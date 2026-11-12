# DSM + AtBSR：全库测试与交叉案例分析

已处理 `results/analysis/longest_row_ratio.csv` 的 2798 条记录：2794 张矩阵完成全部候选和正确性检查，3 张本地文件缺失，`Spielman_k600` 的一个强制 tile 配置超过 32 位偏移限制，导致完整候选测试失败。清单按非零元数量从小到大执行，不按矩阵大小剔除；每个输入、失败、超时和原始输出单独保存。清单和数据文件的位置见 [manifest](../results/model/dsm-atbsr-hybrid/full-suite/manifest.json)，状态和复测审计见 [audit.json](../results/model/dsm-atbsr-hybrid/full-suite/audit.json)。

首轮固定比较：三种 AtBSR 配置（包括默认模型）、一个 CSR DSM、一个 CSR 两阶段、三种 DSM + AtBSR 配置，以及 cuSPARSE CSR_ALG1。每种候选用两个非均匀输入验证直接执行、再用其中一个输入验证 Graph，均与 CPU long-double 参考比较；计时为 3 次预热后随机交错的 9 次测量，每次 Graph 包含 8 次 SpMV。计时前预热，不强制清缓存，不包含预处理或 Plan 创建。每个后端取预设配置中的最低中位数，最佳非混合方案包括 cuSPARSE；不是穷举调优，也不是冷缓存或单次端到端执行结果。每个矩阵首轮只有一个随机次序种子，性能接近交叉点或收益较大的矩阵需要额外独立复测。

在完成完整候选集的 2794 张矩阵中，以“最佳非混合时间 / 最佳混合时间”为加速比：69 张超过 1×，其中 27 张的混合时间至少缩短 5%；2342 张低于 1/1.05×（慢超过 5%）。中位数为 0.809×，几何平均为 0.674×。固定阈值 1024 下，2361 张没有 DSM 长行，其中 23 张混合路径快于对照；433 张有长行，其中 46 张快于对照。微小领先可能是计时波动，也可能来自其他配置差异，不能直接归因于 DSM。这是**优化前冻结版本的固定候选**，不能解释为优化后 Auto Plan 的全库表现。[最终汇总](../results/model/dsm-atbsr-hybrid/full-suite/completion.json)、[逐矩阵结果](../results/model/dsm-atbsr-hybrid/full-suite/analysis.csv)、[分布图](../results/model/dsm-atbsr-hybrid/full-suite/speedups.png)。

对 `Spielman_k600` 单独移除超出偏移限制的强制 AtBSR tile 后，剩余 8 个候选通过正确性检查和计时；最佳混合为 2.724 ms，最佳非混合 cuSPARSE 为 2.494 ms，加速比约 0.916×。由于候选集不同，该结果未并入上面的统一统计。[受限复测状态](../results/model/dsm-atbsr-hybrid/full-suite/limited-recovery/status.json)、[原始计时](../results/model/dsm-atbsr-hybrid/full-suite/limited-recovery/result.csv)。

如检测到其他进程使用相同 GPU，跑批会中断并丢弃受干扰的当前矩阵，待设备连续空闲 30 秒后重试；它不会停止其他用户的进程。检测是轮询，不能完全排除极短外部任务落在采样间隙。监控日志、状态及设备 UUID 存在 [结果目录](../results/model/dsm-atbsr-hybrid/full-suite/)；源码快照和二进制分别存于 `provenance/`，该批次可继续使用原始可执行文件，即使之后的源码优化改变了工作区。

## 交叉案例的初步观察

此前 38 矩阵实验中，`a0nsdsil` 拥有大量短行和 8 条长行，混合路径超过所测最佳非混合；`c-53` 有 43 条至少 1024 非零元的行，且另有 91 条长度 128–1023 的中等行；`rajat20` 的最长行有 29105 个非零元。`c-53` 当前混合实现把中等行留给普通模式 AtBSR，可能形成较长的行内循环和尾部；`rajat20` 最长行只分给一个 cluster，可能限制行内并行。二者是根据数据分布和现有执行方式提出的假设，不能从单次 profiler 采样直接断言因果。

对原 `c-53` 配置的单次 Nsight Compute 探索采样中，融合 kernel 的 achieved occupancy 约为 74%，AtBSR 计算 kernel 约为 41%；对应 L2 hit rate 分别约为 82% 和 88%。它说明退化不能简单归因于“融合路径 occupancy 更低”；采样的两个 kernel 并不包含相同的清零和同步工作，分析器重放及未受控缓存会改变行为。这些计数不能直接与 Graph 的端到端时间相减，也不足以排除尾部或调度开销。[融合采样](../results/model/dsm-atbsr-hybrid/full-suite/crossover-profile/c-53-hybrid_dsm_atbsr-ncu.csv)、[AtBSR 采样](../results/model/dsm-atbsr-hybrid/full-suite/crossover-profile/c-53-atbsr-ncu.csv)。

在排他设备上用 3 个随机顺序、各 21 次 Graph 计时复测，`c-53` 原混合候选为 12.988 μs，阈值 128、C=4、每组 4 行的新候选为 8.032 μs，约快 1.62×；但最佳所测 AtBSR 为 5.508 μs，仍更快。`rajat20` 原混合候选为 18.320 μs，相同思路的新候选为 34.260 μs，不能统一降低阈值。`a0nsdsil` 旧混合候选为 4.220 μs，所测新候选为 9.696 μs。这里只比较指定候选，不能将每种场景的差距全部归因于阈值，因为行分组、线程数和 tile 也同时改变。[隔离复测及配置](../results/model/dsm-atbsr-hybrid/full-suite/optimization-validation-v2/summary.json)；[探索数据和 profiling 口径](../results/model/dsm-atbsr-hybrid/full-suite/crossover-profile/README.md)。

进一步做单参数消融时，固定 `c-53` 的 tile M=4、C=4、256 线程、每组 4 行，阈值 1024 为 21.508 μs，阈值 512 为 12.044 μs，阈值 256 为 8.248 μs，阈值 128 为 8.080 μs；对应的长行数量分别为 43、74、97、134，长行非零元占总数约 20.6%、27.5%、29.5%、31.3%。该对照支持中等行划分是此配置退化的重要因素，但具体是短行尾部、工作平衡还是其他因素，仍需更细的指令级消融。固定阈值 128 和其他参数，将 C 从 4 调至 8 后为 6.160 μs，接近但仍落后同轮 AtBSR 的 5.612 μs。上面的原混合候选与此消融的初始配置不同，不能把 21.508 μs 当作原最佳混合时间。每项为三个随机顺序中位数的中位数，完整参数与结果见[消融汇总](../results/model/dsm-atbsr-hybrid/full-suite/ablation/final/summary.json)。

针对这些观察，混合自动选型在行长度分布产生不同划分时增加阈值 128/256 的候选；相同长行集合与 tile 的候选共享普通行打包。显式指定后端或提供候选列表时保留用户指定配置。新增候选只参与 `tune=true` 实测，默认未调参路径不变。此前复测发现 `ccc` 没有长行，但混合短行路径比最佳所测非混合快约 0.7%；当时曾保留每种 tile 的一个无长行代表配置，三次初始化实测的最佳候选时间中位数从 3.156 μs 变为 3.140 μs，差异很小。[历史初始化对照](../results/model/dsm-atbsr-hybrid/full-suite/optimization-validation-v3/summary.json)。当前 Auto 已移除内置无长行候选，且仅当有 DSM 长行的混合候选比最佳非混合实测时间至少低 5% 时才选择混合；这一新规则仍需全库独立验证。

三次交错初始化实验中，完全剪掉无长行候选的上一版在 `t60k` 上相对最初版本的 Auto Plan 初始化中位数由 753 ms 降至 648 ms；`c-53` 因增加候选由 1394 ms 升至 1512 ms，表明更多实测配置会增加一次性代价。两者计时包含矩阵复制、打包和搜索，结果受环境影响，不是每次 SpMV 的时间，也不能直接作为保留代表候选后版本的开销。加入 C=8 候选后的三次 Auto 实测，`c-53` 均选择 AtBSR；额外候选使其初始化中位数由 1526 ms 增至 1692 ms，仍需在实际多次调用中摊薄。`ccc` 在保留代表候选后均选择混合，但差距只有约 1%。数据量仍不足以宣称选型整体更准确或普遍更快。[最新初始化对照](../results/model/dsm-atbsr-hybrid/full-suite/optimization-validation-v4/summary.json)。

当前源码在 H100 上通过 Plan 快速测试的 280 项直接/Graph 参考检查，以及 CUDA 13 memcheck、synccheck、racecheck；Ada 非 DSM 设备通过 124 项检查。[验证日志及源码哈希](../results/model/dsm-atbsr-hybrid/full-suite/workspace-validation-v4/validation.json)。

## 进度和复现

达到 2000 个有效矩阵时保存了 [2007 矩阵的阶段性快照](../results/model/dsm-atbsr-hybrid/full-suite/snapshot-2000/analysis.json)：固定候选中的最佳混合方案有 48 个快于所测最佳非混合、18 个至少快 5%，1712 个慢超过 5%，加速比几何平均为 0.666×。其中 1783 个最大行小于 1024，224 个达到该阈值。这是按非零元数量排序的前段结果、每个矩阵仅一个随机次序，且基线是有限配置；它不能作为全库或独立复测后的结论。

随后保存的 [2638 矩阵快照](../results/model/dsm-atbsr-hybrid/full-suite/checkpoint-2600/analysis.json)中，有 64 个快于所测最佳非混合、27 个至少快 5%，2214 个慢超过 5%，加速比几何平均为 0.668×；仍有 160 项待处理。该快照仍按规模排序，使用优化前的固定候选，不能作为优化后 Auto Plan 的全库结论。[分布图](../results/model/dsm-atbsr-hybrid/full-suite/checkpoint-2600/speedups.png)。

更新的 [2753 矩阵快照](../results/model/dsm-atbsr-hybrid/full-suite/checkpoint-2700/analysis.json)中，64 个快于所测最佳非混合、27 个至少快 5%，2312 个慢超过 5%，加速比几何平均为 0.672×；清单中还有 45 项待处理。这仍是优化前固定候选的单随机次序首轮结果。[分布图](../results/model/dsm-atbsr-hybrid/full-suite/checkpoint-2700/speedups.png)。

运行 `python3 scripts/analysis/analyze_dsm_atbsr_full.py results/model/dsm-atbsr-hybrid/full-suite` 可更新 [analysis.json](../results/model/dsm-atbsr-hybrid/full-suite/analysis.json) 和 [analysis.csv](../results/model/dsm-atbsr-hybrid/full-suite/analysis.csv)，其中分别保留未完成和失败数量。`runner.log` 是后台运行/等待记录；`status/` 中每个 JSON 对应一个已处理矩阵。测试按非零元数排序，前期样本主要为无长行小矩阵，不能当作总体结论。

已完成样本的可视化见[加速比分布图](../results/model/dsm-atbsr-hybrid/full-suite/speedups.png)；图题记录制图时的样本数量。更新分析后运行 `python3 scripts/analysis/plot_dsm_atbsr_full.py results/model/dsm-atbsr-hybrid/full-suite` 可重画。跑批退出后，后台汇总程序会自动刷新分析和图，并将是否处理完清单写入 `completion.json`；处理完不代表每项都成功，失败与缺失仍单独记录。

可运行 `python3 scripts/analysis/audit_dsm_atbsr_full.py results/model/dsm-atbsr-hybrid/full-suite` 审计首轮每个成功矩阵的候选集合、试次编号与有效计时，并检查三组定向复测的候选集合和原始结果哈希；[audit.json](../results/model/dsm-atbsr-hybrid/full-suite/audit.json) 记录审计时的状态。此检查不能替代 GPU 结果与 CPU 参考值比较，也不验证未完成项。

达到 2000 个有效矩阵后，[交叉点复测程序](../scripts/analysis/benchmark_dsm_atbsr_crossovers.py)会冻结一份包含长行和无长行的初测收益及近交叉案例的清单，再在空闲 H100 上按 3 个随机顺序各测 21 次；此清单依赖初测结果，不是随机留出集。复测覆盖全部 12 种 AtBSR tile/模式、更广的 CSR 配置、cuSPARSE CSR_ALG1 和新低阈值混合配置。程序运行时暂停全库跑批，结束后自动恢复；状态见 `crossover-confirmation/runner.log`。不同阶段的选型和优化结果需要分别报告，不能将它们拼成同一套全库测量。

这组复测已完成 32 个矩阵、96 次独立执行且没有候选校验失败；扩展基线下，12 个矩阵的最佳混合候选快于最佳非混合，其中 8 个快至少 5%。例如 `a5esindl` 为 1.365×、`a0nsdsil` 为 1.349×。这 32 个矩阵按首轮结果有意挑选，不能将 12/32 推广到全库。另一组合配置使 `scfxm1-2b` 的混合时间从 4.844 μs 降至 3.740 μs，但最佳非混合仍更快；该矩阵最大行仅 57，两个配置都没有 DSM 长行，差异涉及 tile、cluster 和线程配置，**不能归因于降低长行阈值**。`c-53` 的改善见上文。无长行的 `ccc` 和 `copter1` 在该复测中分别快约 0.77% 和 0.23%，差异很小。[复测汇总](../results/model/dsm-atbsr-hybrid/full-suite/crossover-confirmation/summary.csv)、[选择与原始运行](../results/model/dsm-atbsr-hybrid/full-suite/crossover-confirmation/manifest.json)。

在 2500 个有效样本后将额外复测中等行案例：取 8 个首轮融合退化最大的最大行 128–1023 的矩阵、从同一层剩余矩阵固定种子抽取 8 个、以及 8 个首轮退化最大的长行矩阵。此试验包含 C=8/阈值 128 的候选，用于检验 `c-53` 的优化是否适用于其他结构；它仍是针对性选择，而不是代表性随机留出集。进度及结果位于 `medium-tail-confirmation/`。

这组 24 个矩阵已完成三轮复测，所有 72 次执行的候选正确性检查通过。扩展非混合对照下，旧固定混合候选在 24 个矩阵中均未获胜；加入新组合后有 10 个获胜，其中 9 个至少快 5%。按预先选取方式分别是：8 个首轮中等行最差案例中 7 个获胜；随机选取的 8 个其余中等行案例中 1 个获胜；8 个首轮长行最差案例中 2 个获胜。该分组只说明这次定向测试的结果，尤其前后两组有意选择了退化案例，不能推广为全库比例。

例如 `freeFlyingRobot_9` 的最大行有 957 个非零元，旧阈值 1024 将其留给普通 AtBSR 短行路径，所测旧混合配置为 34.432 μs；新组合（阈值 128、C=8、256 线程、4 行/组、tile M=4）为 3.676 μs，优于扩展非混合中的最佳 AtBSR 4.392 μs，约 **1.195×**。`rosen1` 分别为 35.192 μs、3.656 μs 和 4.304 μs，新组合相对最佳非混合为 **1.177×**。旧配置仅来自首轮固定的三种混合候选；新组合改变了阈值、分组和可能的 tile，不能将其改善全部归因于阈值。[完整汇总](../results/model/dsm-atbsr-hybrid/full-suite/medium-tail-confirmation/summary.csv)、[选择与原始运行](../results/model/dsm-atbsr-hybrid/full-suite/medium-tail-confirmation/manifest.json)。

对首轮非零元数不少于 100 万、加速比最高的 8 个矩阵也做了相同扩展复测，24 次执行均通过正确性检查；8 个矩阵的最佳所测混合配置均超过扩展非混合对照。例如 `neos3` 为 10.800 μs 对 17.864 μs，约 **1.654×**；`3dtube` 为 12.948 μs 对 17.388 μs，约 **1.343×**。这 8 个是按首轮收益有意挑选的，不能当作大矩阵平均收益。

其中 `nasasrb` 的最大行仅有 276 个非零元：旧阈值 1024 的最佳固定混合候选为 14.588 μs，阈值 128 的新组合为 9.792 μs，优于同轮最佳 AtBSR 的 11.928 μs，约 **1.218×**。这展示了中等行配置可以使本次对比从落后变为领先；不同候选的 cluster 和分组也不同，因此仍不能把收益全部归因于阈值。[大矩阵复测汇总](../results/model/dsm-atbsr-hybrid/full-suite/large-wins-confirmation/summary.csv)、[清单及原始运行](../results/model/dsm-atbsr-hybrid/full-suite/large-wins-confirmation/manifest.json)。

另对 `nasasrb` 交错执行三次旧、新 Auto Plan 初始化：旧版三次均选 AtBSR，选中候选时间中位数为 11.572 μs；优化版三次均选低阈值混合候选，中位数为 10.040 μs，约 **1.153×**。各 Plan 自身采用初始化时的暖 Graph 计时，这不是上段独立候选复测或包含初始化的端到端时间。两版初始化中位数分别约为 1.945 s 和 1.457 s，但仅有三次，受主机和设备状态影响。[Auto Plan 对照](../results/model/dsm-atbsr-hybrid/full-suite/nasasrb-auto/summary.json)。

## 实际选型的后续评测

当前 Auto 实测候选包括 cuSPARSE CSR_ALG1；仅当有 DSM 长行的最佳混合候选比最佳非混合候选的测量时间至少低 5% 时才选择混合。门槛可配置，不保证后续独立执行仍获益。为验证实际选型，新的评测先构建并选定 Plan，再与默认 AtBSR 和预处理后的 cuSPARSE 交错独立计时，每个候选仍做 CPU 参考值和 Graph 正确性检查。计时为暖 Graph，不包含初始化，因此不代表单次端到端收益。

H100 3 号卡上有用户允许共存的常驻进程，初始可用显存约 8 GiB。先对最多 100 万非零元的已完成矩阵分为有、无 1024 阈值长行两层，各固定种子抽取 40 个；这是分层样本，不能直接推断全库比例。[样本清单与结果](../results/model/dsm-atbsr-hybrid/plan-selection-shared-h100/)。随后安排同范围全部 2247 张矩阵的配对实测：[模型选型](../results/model/hybrid-time-model/plan-model-2247-v3/)和[初始化实测选型](../results/model/hybrid-time-model/plan-tune-2247-v4/)。更大的矩阵尚未在这个受限显存条件下验证；设备共享、采样间隙和常驻进程可能影响时序，即使监控到活动时丢弃重测也不能保证完全排除干扰。需要等全部结果和失败项核对完成后才能下结论。

80 张分层样本已完成，实测选型选择混合 15 次；相对默认 AtBSR 的几何平均加速比为 1.151×，相对默认 AtBSR 与 cuSPARSE 中较快者为 1.105×。这属于上述分层样本、共享设备和暖 Graph 口径，包含初始化实测选型但不计其开销，不代表 2247 张总体，也不是后续免实测模型的结果。80 张样本的最初测试未监控常驻进程的逐进程利用率；后续的模型和 2247 张测试增加该监控。[完整结果](../results/model/dsm-atbsr-hybrid/plan-selection-shared-h100/analysis.json)。

后来发现创建顺序是上述测量的重要混杂因素：在 H100 上将两个配置完全相同的默认 AtBSR Plan 依次创建并比较，`n2c6-b10` 的先建/后建时间比约 0.963；交换创建顺序后，同一标签时间比约 1.042。`lp_kb2` 也出现约 0.960 和 1.042。原因可能与分配地址或缓存布局有关，尚未定位；因此此前固定顺序的微小差距、总体加速比及其训练标签都可能有偏，不能作为严格的性能确认。新测试会对每个矩阵交换 Plan 创建顺序，分别计时后对两种顺序的时间比取几何平均；它仍不能消除共享设备或缓存状态带来的全部变化。[同配置对照原始数据](../results/model/hybrid-time-model/baseline-control-h100/)。

当前代码在该 H100 上通过 328 项直接/Graph 参考检查，并通过 CUDA 13 的 memcheck、synccheck 与 racecheck；CUDA 12.4 memcheck 对 DSM shared 访问报错，尚未确定与 CUDA 13 结果不一致的原因，不将其视作已解决。[CUDA 13 验证日志](../results/model/dsm-atbsr-hybrid/plan-selection-shared-h100/h100-memcheck-cuda13.log)。
