# 100 矩阵配对对照

本轮入口为 `h100-20261111/`，最终结论见 [报告](../../../docs/spmv-sample100.md)。

## 清单

`select_spmv_sample.py` 仅使用结构和输入头部选择样本，不读取新旧性能：四个 nnz 档各 25 个，每档 13 个最长行小于 1024 的矩阵、12 个含至少 1024 非零元长行的矩阵；名称去数字后的近似家族最多 3 个。排除此前 24 个已反复分析的矩阵。限制实数/整数/pattern coordinate、行列均不超过 200 万、nnz 在 1000–2000 万范围，输入文件必须存在。名单在新计时前写入 manifest，之后不替换失败或退化案例。它不是全库随机抽样，也不保证与模型训练数据独立。

## 候选

- `atbsr_control`：原 AtBSR shape/均衡选择。
- `atbsr_grouped_model`：同一选择，尝试启用六种普通 shape 的分组列号；不满足条件时回退。
- `auto_refined_v2`：Auto、vector v2 模型、有界初始化实测、mixed 候选；保持既有收益门槛。
- `auto_refined_v2_grouped`：上一方案允许 AtBSR 候选启用分组列号；仍以初始化实测决定赢家。
- cuSPARSE CSR ALG1、DEFAULT、ALG2：三者各自构建与 preprocess。

有界精调只比较已有模型候选、默认 AtBSR、六种 CSR vector 和五种 mixed 实现，不是全量穷举所有 DSM/AtBSR 参数。模型范围外仍执行有界实测回退。新实验标签在 `bench/detail/spmvBenchmarkOptions.h` 映射到已有生产 API，没有修改生产选择算法。

## 协议与文件

两个独立进程反转 Plan 构建顺序，FP64、prefetch=2、4 warp/CTA、cap=8；不设置 persisting L2。每进程 3 轮预热、9 轮随机交错测量，每个 Graph batch 连续 8 次 SpMV，Graph 内 event 计时。每候选对两组非均匀 X 和第二组的 Graph 输出做 CPU long double 全行参考检查。主速度比不含读取、格式转换、调优和 Graph 构建；setup 单列。不同顺序可以选中不同 backend，原始 CSV 保留实际配置。

`manifest.json` 固定输入、分层及候选；`metadata.json` 固定二进制/驱动/manifest 哈希和 GPU UUID；`raw/` 保存每进程计时与日志；`status/` 保存输入哈希、命令、退出码及干扰/超时状态；`provenance/` 保存对应版本的源文件和二进制。检查失败项，不能仅因 completed 文件存在就当作全部成功。

`comparison.csv` 列出每个方案相对每进程最快 cuSPARSE、相对 ALG1、setup 和实际配置；`comparison-summary.json` 汇总全样本和分层结果。所有几何均值均先配对两个进程的时间比，随后跨矩阵取几何平均。最快 cuSPARSE 是实测三者中的最快，不是该库所有可能配置的全局最优。

```bash
python3 scripts/analysis/select_spmv_sample.py --output /tmp/new-sample100 --gpu 3
# 测试时 GPU 应计算空闲，脚本监测到其他进程会作废该矩阵的本轮尝试。
python3 scripts/analysis/benchmark_optimization_ablation.py \
  --output results/model/spmv-sample100/h100-20261111 --exe /tmp/atspmv-sample100-1111
python3 scripts/analysis/analyze_spmv_sample.py results/model/spmv-sample100/h100-20261111
```

使用新清单应另建输出目录。恢复旧测量时必须使用 metadata 对应的冻结驱动和二进制；改代码后不要覆盖原目录。恢复后后台会话为 `atspmv_sample100_nvml_1111`，本轮已全部结束。可用 `tail -n 10 h100-20261111/runner.log` 查看状态；VS Code 断开不影响 tmux。

监控迁移记录见 `h100-20261111/runner-migration.json`：前 58 个成功矩阵使用原查询，后 42 个使用 NVML。各矩阵状态保存对应 runner 哈希，二进制和计时协议未变。
