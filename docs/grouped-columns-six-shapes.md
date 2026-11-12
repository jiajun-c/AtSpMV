# 分组列索引扩展到六种普通 tile shape

已将 `group_tile_columns` 从普通 32×1 推广到六种普通 shape；balanced、已有 int16 列号和不省空间的情况继续回退。默认开关保持关闭。扩展修改计算映射和数值更新，不改变矩阵值精度，不使用 Tensor Core、DSM 或 X 的 L2 保留窗口。

## 实现

- 分类、三个前缀扫描、按编码宽度重排及存储成本判断复用原实现：每个 tile 仍为 32 个槽。
- `groupedTileSpmv<M>` 使用 K=32/M，将三个索引池的 lane 部分和按 K 做 shuffle 归约，仅 `lane%K==0` 写回 `order[group*M+lane/K]`。M=32 时没有额外归约。
- 数值更新按 `row=order[source_group*M+lane/K]`、`position=source_tile*K+lane%K` 回读原 CSR；填充槽写零，来源映射复用，不重新排序或分组。
- [groupedTileColumns.cuh](../include/detail/groupedTileColumns.cuh) 实现六个编译期特化；[spmvPlan.cu](../src/spmvPlan.cu) 传入 M 进行更新。API 沿用现有开关，没有新增必填参数。

```cpp
atspmv::SpmvPlanOptions options;
options.backend = atspmv::SpmvBackend::AtBsr;
options.tile_index = 3;  // 0..5: 1x32, 2x16, 4x8, 8x4, 16x2, 32x1
options.balanced = 0;
options.group_tile_columns = true;
options.enable_value_updates = true; // 仅需要更新数值时开启
auto plan = atspmv::makeSpmvPlan(device_csr, options, stream);
// plan.info().tile_columns_grouped 表示实际生效；仍可能回退。
```

设 tile_index=-1、balanced=-1 时，保留原 AtBSR 的 shape/均衡选择，再尝试分组；模型尚未为分组布局标定，不会预测并比较压缩开/关的最优组合。旧 `compress_tile_columns` 仍只支持普通 32×1，与新开关互斥。

## 正确性

- 更新测试 138 组全部通过，覆盖三种列维度、六种普通 shape、balanced/int16 回退、空行和不完整行组、不同编码池、共享调优存储及直接更新/Graph 重放。
- 压缩/分组专项 57 组在 CUDA 12.6 memcheck 下通过，`ERROR SUMMARY: 0 errors`；没有将原 DSM 路径包含在此次 sanitizer 结论中。
- 24 个真实矩阵的全部候选均通过两个非均匀 X 的直接结果和 Graph 结果检查，对照 CPU long double 全行参考。共 816 个候选实例、2448 次全向量检查。

## 同 shape 性能对照

在同一张空闲 H100 PCIe GPU 3 复测此前固定的 24 矩阵。每矩阵六种普通 shape 各测未分组/分组，再测原自动 AtBSR、开启分组的自动 AtBSR，以及 cuSPARSE DEFAULT/CSR ALG1/CSR ALG2，共 17 个候选。两个独立进程反转构建顺序，每进程 3 轮预热、9 轮随机交错计时，每 batch 8 次 Graph SpMV，采用内部 event。报告进程中位时间比的几何平均，共 7344 条有效计时。

两种布局均采用两路累加对照（原 kernel prefetch=2），4 warp/CTA，不设置 persisting L2；grouped kernel 固定两路累加。FP64 精度相同，未锁时钟。时间不包含读取、转换、分类、候选构建和 Graph 创建，属于重复调用的稳态性能。

下表只统计实际启用分组的矩阵，回退未计为布局收益；每种 shape 的合格样本不同，不能横向认定某 shape 的绝对性能最好。加速比 = 相同 shape 未分组时间 / 分组时间。

| shape | 实际启用/24 | 几何加速比 | 收益超过1% | 退化超过1% |
|---|---:|---:|---:|---:|
| 1x32 | 12/24 | 1.011× | 6 | 4 |
| 2x16 | 13/24 | 1.080× | 9 | 3 |
| 4x8 | 15/24 | 1.112× | 12 | 2 |
| 8x4 | 18/24 | 1.133× | 13 | 5 |
| 16x2 | 21/24 | 1.105× | 14 | 4 |
| 32x1 | 21/24 | 1.112× | 17 | 4 |

六种 shape 都存在受益案例，但 1×32 在本组整体基本持平。较小 M 的每行组区间元数据更难摊薄，部分矩阵直接因不省空间而回退；不同 M/K 也会改变 tile 内列跨度、任务长度及编码比例。不能仅由索引字节减少推导运行时间收益。

| 矩阵 | 1×32 | 2×16 | 4×8 | 8×4 | 16×2 | 32×1 |
|---|---:|---:|---:|---:|---:|---:|
| FEM_3D_thermal2 | 0.993× | 0.954× | 1.077× | 1.532× | 1.663× | 1.909× |
| pwtk | 0.965× | 1.097× | 1.166× | 1.241× | 1.286× | 1.341× |
| rma10 | 0.990× | 1.002× | 1.072× | 1.068× | 0.884× | 0.885× |
| mac_econ_fwd500 | 回退 | 回退 | 0.917× | 0.914× | 0.977× | 0.891× |

## 保留原选择后开启分组

全 24 矩阵相对关闭开关的几何加速比为 **1.088×**；15 个收益超过 1%，3 个退化超过 1%，其余基本持平。实际启用的 19 个矩阵为 **1.113×**，其余回退。明确退化包括 pdb1HYS（0.952×）、rma10（0.884×）、mac_econ_fwd500（0.877×）。因此支持更多 shape 并不意味着应默认全部开启。

相对每进程测得的三种 cuSPARSE 中最快者，开启分组的自动 AtBSR 在本样本为 **0.944×**，11 个收益、13 个退化超过 1%，整体仍慢于 cuSPARSE。

事后从六种分组普通 shape 中挑最快，对照六种未分组普通 shape 中最快者为 **1.115×**；它没有包含 balanced、DSM、mixed 等候选，也不是模型实际选出的结果，不能当成线上统一选择器的收益。

本轮和此前仅支持 32×1 的 24 矩阵实验使用不同候选集合与独立分配。不能用两轮均值相减，断言扩展本身带来额外总体加速；本轮证据支持“新增 shape 的分组路径可正确执行，并在部分相同 shape 对照中更快”。反序构建也不能消除所有地址、缓存和 Graph 上下文影响，暂不据此修改默认开关或原时间模型。

## 数据与复现

- [按 shape 的汇总](../results/model/grouped-columns/h100-six-shapes-20261111/shape-summary.json)
- [逐矩阵×shape 的时间及两种顺序比值](../results/model/grouped-columns/h100-six-shapes-20261111/shapes.csv)
- [保留原自动选择的对照](../results/model/grouped-columns/h100-six-shapes-20261111/matrices.csv)
- [输入、候选与采样清单](../results/model/grouped-columns/h100-six-shapes-20261111/manifest.json)
- [正确性日志](../results/model/grouped-columns/h100-six-shapes-20261111/value-update.log)；[memcheck](../results/model/grouped-columns/h100-six-shapes-20261111/memcheck.log)

```bash
python3 scripts/analysis/analyze_grouped_shapes.py \
  results/model/grouped-columns/h100-six-shapes-20261111
```

原始 CSV/log 在同目录 raw/，进程退出状态在 status/，二进制、源快照及身份哈希在 provenance/、metadata.json。跑批继续使用 benchmark_optimization_ablation.py，计时协议与 [消融目录说明](../results/model/optimization-ablation/README.md) 一致；本轮已全部结束。历史布局结果和全优化清单见 [优化报告](optimization-report.md)，不将其旧数字覆盖为新版结果。
