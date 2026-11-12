# 每个 16×16 块只有一个非零元的 SpMM 实验

测试矩阵 `A` 为 1024×1024，按 `16×16` 块划分，共 64×64 个块位置；每个选中的块在随机的一个位置存储一个非零元，其余 255 个位置为零。严格全块版本的 4096 个块位置均被选中，因此共有 4096 个非零元，矩阵整体密度与块内密度均为 1/256。另设每块行 1、2、8、32 个非空块的版本，其未选择的块完全为空，分别有 64、128、512、2048 个非零元。位置和输入使用固定种子。生成器逐块检查非零元数，并单独核对 CSR 与 Blocked-ELL 表示。

以 cuSPARSE CSR ALG2、CSR ALG3 和 Blocked-ELL ALG1 为候选，B/C 为行主序；A/B 为 FP16，C 与计算为 FP32。每种配置测试 N=8、32、128，各测试正向和反向构建顺序，共 30 次，逐输出与 CPU 参考验证。计时为预热 Graph，不含格式转换、初始化及 Graph 构建；不能直接推断 FP64 或端到端性能。测试完成后还对部分配置进行 Nsight Compute Tensor、HMMA、FFMA 和 HFMA 计数，以实际计算 kernel 而非格式名判断使用的指令；CSR ALG3 已知可能因输入不同而使用 Tensor 指令。

任务使用 `atspmv_spmm_exact_one` tmux 会话，在 3 号 H100 连续空闲时启动；采样式干扰检测不能排除全部外部影响。实时状态见 [runner.log](../results/model/cusparse-spmm-core-compare/h100-exact-one-v1/runner.log)，结果及配置保存在 [实验目录](../results/model/cusparse-spmm-core-compare/h100-exact-one-v1/)。完成后 `comparison.csv` 和 `summary.json` 汇总两种顺序，`instruction-profile/` 保留原始指令采样。中断或有其他 GPU 进程时自动等待和重试，失败项不会视为成功结果。
