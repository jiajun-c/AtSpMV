# 混合 tile 性能分析：Ada 代表案例

使用 RTX 5880 Ada 对四张矩阵采集 Nsight Compute 2025.3 和部分 Nsight Systems 时间线。全库扫库在采集期间暂停，随后恢复；基线和混合都在同一 GPU 上串行采集。选择的配置来自既有 paired Graph 测试：`trans4` 来自 80 张对照，其他三张来自全库中途汇总。样本有意覆盖加速与变慢，不代表总体。

独立 profile 入口构建相同候选，执行 10 次预热，然后用 CUDA profiler API 标记一次直接调用；输入向量为零。NCU 启用 `--cache-control none --clock-control none`，L2 可保留预热结果，但缓存状态和 GPU 时钟未固定；指标来自多次 replay，可能受波动影响。两组 CSV 是不同采集轮次。profile 捕获的直接调用与原实验的 8 次一组 warm Graph 不同，计时不可直接相减。NCU 只列出 kernel，不包含 mixed 的 `cudaMemsetAsync`；balanced AtBSR 的 `beta_vecY_prod` 是 kernel，单列。Nsight Systems 记录了直接调用的 kernel 和 memset，同样不是 Graph 时间。GPU sector 和执行指令不计 CPU 工作、图调度和矩阵预处理。

| 矩阵 | 混合 / 最优固定槽数 | Graph 最快固定时间 | Graph 混合时间 | Graph 加速比 | 所用配置 |
|---|---:|---:|---:|---:|---|
| trans4 | 0.503 | 11.200 μs | 6.848 μs | 1.636× | m8_b1_cap16 vs mixed_bal8 |
| bloweybl | 0.500 | 3.840 μs | 3.072 μs | 1.250× | m8_b1 vs mixed_bal4 |
| rajat09 | 0.974 | 3.072 μs | 2.304 μs | 1.333× | m32_b1 vs mixed |
| Chebyshev1 | 0.569 | 2.300 μs | 3.454 μs | 0.666× | m1_b0 vs mixed_bal4 |

## 观察

- **trans4：减少填充与工作量下降同时出现。** NCU 的 SpMV 主 kernel：混合 L2 请求 667,256 sector，固定方案 968,264 sector，减少约 31%；混合执行 567,768 条 warp 指令，固定执行 1,630,468 条，减少约 65%。混合 grid 1,205 block，固定 grid 4,102 block。两者的 L2 hit rate 分别约 98.0% 和 99.8%，DRAM 读取都很少。结果支持混合减少 L2 访问及 warp 工作量的解释；但两个方案 shape 和任务划分也不同，不能把全部差距归因于填充。
- **bloweybl：混合主 kernel 请求 81,469 L2 sector，固定为 108,319，减少约 25%；warp 指令从 347,245 降到 119,175，减少约 66%。** 固定 balanced 另有 beta kernel；混合也要 memset 输出，NCU kernel 表没有计入后者。此结果支持填充和任务分解减少工作量的解释，但不能独立区分各自贡献。
- **rajat09：收益主要与执行路径相关，密度改善很小。** 两个 SpMV 主 kernel 的 L2 sector 分别为 70,662 和 71,140，差距小于 1%；混合反而执行更多 warp 指令（117,619 对 86,203）。混合非 balanced 直接写回，不需要先清零；固定 balanced 先执行 beta kernel、再执行 SpMV。Nsight Systems 的一次直接调用看到混合 1 个 kernel，固定 2 个 kernel，支持减少初始化/launch 是收益来源之一，但不证明其全部加速比例。
- **Chebyshev1：密度提升却整体变慢。** 混合的 SpMV 主 kernel 使用 7 个 block、2,394 条 warp 指令，固定为 66 个 block、21,518 条；但混合的 L2 sector 为 7,570，固定为 6,385，且混合必须 memset 和 atomicAdd，固定 unbalanced 直接写回。Nsight Systems 一次直接调用中混合有一次 memset 和一次 kernel，固定仅一个 kernel。小任务上的固定开销与额外访问是合理候选解释；NCU 和单次时间线并未复现 Graph 的全部时间差，尚不能定量证明根因。

硬件计数的 L2 sector 包含多种访问和其他请求，并非仅矩阵值流量；每轮为单次采样，不能当作稳定置信区间。DRAM 在这些预热案例中很少，不能据此推断冷缓存或大工作集。结果也不适用于 H100。

原始数据、重放脚本及 profiling 可执行文件和源码快照位于 [profile-ada](../results/model/mixed-tile/profile-ada/)。全库任务使用原有独立二进制，未替换。若需进一步确认因果关系，应在相同任务切分与输出汇总方式下做消融，对比只改变布局的两个 kernel，再在多轮 Graph 测量中复测，并扩展更大工作集。
