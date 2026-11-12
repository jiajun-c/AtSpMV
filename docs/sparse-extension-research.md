# 从 AtSpMV 扩展到其他稀疏算子

研究日期：2026-11-02。本文是候选方向与实验计划，不是已实现的 SpMM/SDDMM 性能结论。当前仓库尚无这些算子的同口径性能结果，已有 SpMV 的 H100 加速比不能迁移到它们。

## 可复用与不可直接复用

可以复用 CSR 合法性检查、行长统计与排序、按行分组、长行拆分、Plan 生命周期、资源管理、模型与实测选择框架、数据集及正确性测试方法。跨 CTA 规约和 DSM 代码可作为设计参考，但必须根据输出维度重新设计。

AtBSR 的 `m×k=32` tile 是按每行非零元的**存储序号**分组，列索引仍独立保存；它不保证同一 tile 内的行共享对齐的列块，因此不是 Tensor Core MMA 所需的稠密矩阵块。SpMM 的稠密输入 `B` 还有特征维度 N、行/列主序及跨列并行和复用问题；SpMV 的代价参数、候选、时间标签和形状都不能直接套用。

## 优先方向：小到中等特征宽度的 FP64 CSR SpMM

对于 `C=A×B`，先聚焦无结构 CSR、固定 A 多次复用、FP64 和明确的 row-major B/C（先支持非转置，`alpha=1,beta=0`；扩展语义单列）。这与现有科学计算数据和 FP64 SpMV 最一致，能够先研究选择器和行不均衡，而不依赖降低精度。优先测 N=1、2、4、8、16、32、64，并包含 31、33 等非整 tile 宽度。N=1 需要与现有 SpMV 对照，检查额外开销。

候选：

1. CSR vector：线程在行内非零元与输出列之间二维分配，跨列复用 A 的值和索引；针对窄 N 设计子 warp 映射。
2. AtBSR CUDA core：沿用行排序、非零元分组和列索引 gather，新增输出列维度分块；不要把原 tile 当作 MMA tile。
3. 长行拆分：按列块产生部分结果并规约，比较普通 CTA、全局两阶段与 DSM cluster；DSM 的同步及共享内存成本随 N 增加。
4. 可选高密度块 Tensor Core：独立构建真实的列对齐块、记录 padding、索引映射和 B gather；同时保留 CUDA core 回退。数据类型、累加精度和允许误差必须与被比较的基线一致。是否值得使用 FP64 Tensor Core 需要实际测量，不能假定总是更快。

先建立同一 `SpmmPlan` 的有限候选和实测 oracle，再增加模型：特征包括行长度分布、AtBSR 填充率、列索引重用/块密度、N、数据类型、布局、设备资源；目标直接估计各候选耗时和初始化成本，按预计复用次数选择。模型仅在独立冻结的按矩阵族划分数据上评估，并保留 cuSPARSE 回退。候选和阈值需事先冻结，避免在测试集上调优。

**可检验的收益假设，不是结论：** 窄 N 且行长跨度大时，合适的子 warp 宽度、长行拆分和跨列复用可能优于单一固定映射；较高列块密度且 N 足够大时，重排或稠密化成本可摊薄后，Tensor Core 路径可能胜出。无结构低密度、稀疏索引重用少或 N 很小时，填充和 gather 可能抵消矩阵乘法吞吐优势；单次调用也可能被准备成本主导。

## 基线与验证

官方基线使用 [cuSPARSE SpMM](https://docs.nvidia.com/cuda/archive/12.4.1/cusparse/index.html#cusparsespmm)（如链接锚点变化，可从[官方 API 文档](https://docs.nvidia.com/cuda/cusparse/)查找 `cusparseSpMM`），覆盖该版本实际支持的 CSR 算法及布局，分别记录有无预处理、工作区和初始化时间；符合数据格式时再比较 Blocked-ELL/Tensor Core。使用相同 `alpha`、`beta`、转置、数据类型和容差。将版本、时钟、设备、数据集哈希和布局记录在清单中。

研究对照至少包含 [RoDe](https://github.com/CRAFT-THU/RoDe)（CUDA core）、[FlashSparse](https://github.com/ParCIS/FlashSparse)（Tensor Core，提供 SpMM/SDDMM 和 H100 复现实验）及 [Acc-SpMM](https://github.com/Hyaloid/AccSpMM)（Tensor Core，列出 H100 支持）。实际运行前逐一核对版本、许可证、输入格式、支持的精度和输出语义；不支持 FP64 的实现不能作为 FP64 同精度性能对照，单独列为不同精度实验。[Acc-SpMM 论文](https://arxiv.org/abs/2501.09251)报告其 H100 上对 cuSPARSE 的平均加速，但该数值由其自身数据与口径得到，不能当作本项目结果。[cuTeSpMM](https://arxiv.org/abs/2504.06443)也指出 Tensor Core 效果依赖稀疏结构与计算映射的匹配程度。

每张矩阵覆盖短行占优、长尾、块结构和随机列索引等层次，以及 N、精度、布局和复用次数；不支持或失败案例必须保留。CPU 高精度参考检查空行、重复列、非对齐 N、非零 alpha/beta 和数值误差，检查实际候选输出。两种构建顺序、随机交错多次测量，分别给出冷/暖缓存、纯 kernel、Plan/预处理，以及 `T_total(R)=T_prepare+R*T_execute`。报告逐矩阵比值、几何均值、中位数、尾部退化、距同轮 oracle 的差距及选中频率；对多种算法比较时要先说明哪一方承担选择成本。不得把跨论文或跨 GPU 的数字当作同轮对照，也不能只报告筛选后的收益案例。

## 其他算子

| 方向 | 复用机会 | 主要新增问题 | 优先级 |
| --- | --- | --- | --- |
| 多右端 SpMM | CSR、行长度统计、分组与 Plan | B/C 特征维度、布局、列重用及新 kernel | 高 |
| SpMV + 向量更新/点积 | 现有 SpMV 和求解器应用 | 同步、规约、数值顺序及端到端语义 | 高；适合固定求解器验证 |
| SDDMM | 固定稀疏结构、边分组及任务调度 | 每个非零位置的稠密向量点积；输出按稀疏结构写回 | 中；需新 kernel |
| 稀疏 attention 融合 | SDDMM/SpMM 结构与行分组 | 行 softmax、在线归一化、保存中间结果与反向传播 | 后续；整链竞争激烈 |
| SpGEMM/SpTRSV | 输入检查与部分调度设施 | 输出结构生成/去重，或行依赖及拓扑调度 | 低；无法直接复用 SpMV kernel |

[cuSPARSE SDDMM](https://docs.nvidia.com/cuda/archive/12.4.1/cusparse/index.html#cusparsesddmm) 文档定义了对稀疏结构采样的稠密矩阵乘法及可选预处理；实施时以实际版本文档为准。[Fused3S](https://arxiv.org/abs/2505.08098) 已研究融合 SDDMM、softmax 和 SpMM，[Libra](https://arxiv.org/abs/2506.22714) 已研究 CUDA Core 与 Tensor Core 联合调度，因此单纯提出融合或混合硬件不是足够的创新或性能证据。

建议先完成 FP64 窄 N 的可复现对照与失败分类，再根据实测挑选有收益的矩阵族设计 kernel 和模型；在没有对照之前，不声称优于官方或现有方法。
