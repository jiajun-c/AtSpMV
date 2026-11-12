# 利用分布式共享内存优化稀疏矩阵向量乘：背景与出发点

稀疏矩阵向量乘（SpMV）是迭代求解、图算法和科学计算中的基础操作。对于 CSR 格式，输出行 $i$ 可以写成

$$
y_i = \sum_{j=\operatorname{rowptr}[i]}^{\operatorname{rowptr}[i+1]-1} A_j x_{\operatorname{col}[j]}.
$$

稀疏矩阵的每行非零元数量可能相差很大。当行数较少而少数行很长时，将一行交给一个 CTA 虽然避免了跨 CTA 规约，但可用的 CTA 数量受行数限制，GPU 并行度可能不足；长行还可能形成执行尾部。将一行拆给多个 CTA 可以增加并行工作，但这些 CTA 必须合并各自的部分和。常见办法包括向全局输出做原子加，或先将部分和写入全局内存、再由另一个 kernel 汇总。前者可能产生同一输出地址上的原子竞争，后者引入中间结果访问与额外 kernel。实际代价取决于行长度、行数、缓存状态和并行度，不能预先认定哪种方式一定最优。

## 分布式共享内存提供了新的跨 CTA 通信路径

NVIDIA Hopper 引入 thread block cluster，同一 cluster 内的 CTA 可以协作，并访问彼此分配的共享内存，这种跨 CTA 的地址空间称为 distributed shared memory（DSMEM）。每份共享内存仍由各 CTA 分配，远端 CTA 的访问通过片上互连完成；DSMEM 不是 L2 cache 中的一块新内存，也不自动扩大单个 CTA 的共享内存配额。使用者需要保证目标 CTA 及其共享内存在访问完成之前保持有效，并正确处理初始化、可见性和同步。[NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#distributed-shared-memory)

这一能力为长行规约提供了介于 CTA 内共享内存和全局内存之间的通信范围：多个 CTA 可以并行处理同一行，并将较小的部分和交换和汇总安排在 cluster 内，最终由一个 CTA 写出结果。其潜在收益来自两方面：拆行增加可用并行度；片上通信可能降低部分和汇总所需的全局原子操作或额外 kernel 成本。收益仍须覆盖 DSMEM 访问、同步、通信量和 cluster 占用带来的开销，也不能消除对矩阵数据和向量 $x$ 的读取。

## 相关工作与本项目的设计启发

[QuACK 的 `cluster_reduce`](https://github.com/Dao-AILab/quack/blob/77a750c0d9548d4a6e23c0f64062d61b3ea420ff/quack/reduce.py#L114) 和 [`store_shared_remote`](https://github.com/Dao-AILab/quack/blob/77a750c0d9548d4a6e23c0f64062d61b3ea420ff/quack/utils.py#L37) 展示了一种适用于跨 CTA 规约的实现：先在较小的线程范围内计算部分结果，再利用远端共享内存写入和 `mbarrier` 管理跨 CTA 数据到达。计算、通信与同步可以共同组织，使部分和不必先写回全局内存。我们借鉴的是该异步通信机制，而不是将 QuACK 对归一化等算子的性能直接套用到稀疏矩阵。

[ClusterFusion](https://arxiv.org/abs/2508.18850) 则从算子融合出发，将 DSMEM 通信组织为 `ClusterReduce` 和 `ClusterGather`，让 cluster 内多个 CTA 协作完成规约、数据汇聚和后续计算。论文在其 H100 LLM 推理评估中报告，相对所选推理框架平均端到端加速 1.61×；这一结果属于其模型、算子融合和基线配置，不能视为 SpMV 的预期加速比。论文还指出 cluster size 增大可能降低可用通信带宽和活跃 SM 数，强调通信范围与并行度之间需要权衡。[论文全文](clusterFusion.pdf)

两项工作共同启发我们：**把跨 CTA 的规约作为明确的片上协作问题设计，并结合任务划分、通信方式和同步开销进行评估**。针对 SpMV，我们选择先研究长行，因为其输出天然需要合并多个部分和；不同于 ClusterFusion，当前工作并未实现通用的 cluster collective 库或跨算子融合框架。

## H100 上的通信延迟观察

我们在 H100 PCIe 上构造单线程、串行依赖的 32 位读取链，对比远端 DSMEM、热全局内存和 512 MiB 随机工作集的全局内存访问。每次访问的地址由前一次读取结果决定；表中为三组随机种子各自中位数的中位数。

| cluster size | 远端 DSM，rank 1 | 远端 DSM，最后 rank | 热全局读取 | 大工作集全局读取 |
| --- | ---: | ---: | ---: | ---: |
| 2 | 186.48 cycles | 186.48 cycles | 272.39 cycles | 656.14 cycles |
| 4 | 206.48 cycles | 194.98 cycles | 287.29 cycles | 669.47 cycles |
| 8 | 206.48 cycles | 201.48 cycles | 287.30 cycles | 670.53 cycles |
| 16 | 213.48 cycles | 198.48 cycles | 278.62 cycles | 669.28 cycles |

在这一微基准中，rank 1 的远端 DSMEM 读取约为 186–213 cycles，低于热全局对照的 272–287 cycles，以及大工作集对照的 656–671 cycles。独立 Nsight Compute 采样在大工作集读取中观察到大量 L2 miss 和 DRAM 流量，支持它以 HBM 读取为主，但不能保证计时中的每次读取都来自 HBM。上述数值包含循环和地址计算，地址布局也并不完全相同；它们说明片上跨 CTA 通信具有潜力，**不能直接换算成 DSMEM 的峰值带宽、规约延迟或 SpMV 加速比**。cluster size 16 是非可移植配置，需要设备支持和显式 opt-in。[实验方法与限制](performance-model.md#h100冷全局内存与-dsm-依赖读取)；[原始汇总](../results/model/dsm-hbm-latency/summary.csv)。

## 针对长行 SpMV 的实现与性能

本项目的 [DSM 长行实验实现](../include/dsmLongRow.cuh) 将一条 CSR 行的非零元区间划分给同一 cluster 内的多个 CTA。各 CTA 读取各自负责的矩阵元素和 $x$，计算 FP64 局部乘加，并以 warp 为单位形成部分和；随后通过异步远端共享内存写入和 `mbarrier` 完成部分和交换，rank 0 读取本地接收缓冲区并写出 $y_i$。当前实现保留 all-to-all 发送，最终仅 rank 0 写结果；它并非最小通信量方案。矩阵和 $x$ 仍从全局地址读取，DSM 只用于部分和通信。

在 H100 PCIe 的单次 FP64 SpMV 实验中，我们将异步 DSM 与经过各自配置搜索的 AtSpMV、cuSPARSE、单 CTA、atomic 和两阶段规约对照。下表为三个测量种子中位数的汇总，时间为 CUDA Graph 内每次调用的 GPU 时间；不包含行分类、矩阵格式转换、预处理或 Graph 创建。

| 测试用例 | DSM | 最佳非 DSM | DSM 加速比 |
| --- | ---: | ---: | ---: |
| 64 行、每行 16384 个非零元 | 7.752 μs | 8.344 μs，AtSpMV | 1.076× |
| 32 行、每行 32768 个非零元 | 8.184 μs | 8.756 μs，两阶段 | 1.070× |
| 128 行、每行 8192 个非零元 | 7.368 μs | 7.780 μs，AtSpMV | 1.056× |
| 长短行混合的 137 行案例 | 6.508 μs | 6.192 μs，AtSpMV | 0.951× |

前三项是局部性较好的受控合成 CSR 用例，密度约为 25%，不应视为天然高度稀疏的真实图。它们说明在少行、长行且划分较均匀时，拆行与片上规约的组合可以超过所测最佳非 DSM 方案约 5.6%–7.6%；这一差异同时包含并行分工、规约组织和同步协议变化，不能全部归因于 DSMEM 读取延迟。

在结构筛选的真实宽矩阵 `bibd_19_9` 上，DSM 为 17.248 μs，优于最佳 AtSpMV 的 18.960 μs，约 1.099×；但最佳非 DSM 两阶段规约为 17.176 μs，略快于 DSM。该组八个真实宽矩阵均未稳定超过各自的最佳非 DSM 对照。原先测试的七个不规则真实矩阵也未胜过最佳非 DSM，因此目前不能宣称 DSM 长行路径对真实稀疏矩阵普遍有收益。[完整实验、配置与验证](performance-model.md#单次-spmv-的长行-dsm-规约)。

对于单次 SpMV，将完整 $x$ 放进 DSM 的既有实验也未发现收益：加载、同步、占用和远端访问的成本可能抵消复用带来的好处，而全局读取可能已命中 L2。另一方面，仓库的多轮融合 PageRank 实验在 `NotreDame_yeast` 上报告了相对最快持久化 global 约 1.32× 的收益；这是特定图和多轮迭代的结果，不能归因于单次 SpMV，也不能推广到所有图。[PageRank 实验](performance-model.md#持久化-pagerank-的-dsm-实验)和[完整结果](../results/model/dsm-pagerank-holdout/summary.json)。

这些结果支持继续针对**少行长行、部分和竞争明显、每个 cluster 有足够工作量**的场景探索 DSMEM，也说明必须考虑行长度不均衡、cluster 占用和额外同步。当前统一 Plan 提供可选初始化实测来比较 DSM 与其他候选；默认不凭上述微基准自动假定 DSM 更快。新增长短行混合方案在部分真实数据集上获得收益，也存在明显反例；具体方法和证据见 [DSM + AtBSR 混合实验](dsmem-hybrid.md)。
