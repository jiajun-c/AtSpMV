# AtSpMV 优化消融实验

主报告：[优化与消融报告](../../../docs/optimization-report.md)。本目录只组织本轮新实验，不移动或覆盖历史 DSM、mixed、选择模型及 MFEM 数据。

| 目录 | 实验 | GPU | 覆盖 |
|---|---|---|---|
| `h100-20261111/` | shape/拆分/预取/索引/CTA 参数/mixed/DSM | H100 GPU 3 | 9 矩阵，380 对候选 |
| `h100-row-order-20261111/` | 相同 host packer 与生产 kernel 的行序开关 | H100 GPU 2 | 4 矩阵×2 shape，含 cuSPARSE 共 28 对候选 |

两个目录各自包含：

```text
manifest.json          矩阵、完整候选参数、采样数与设备号；选定后冻结
metadata.json          manifest/驱动/二进制哈希、设备 UUID、公共参数与计时范围
hardware.txt           GPU 型号、UUID、驱动版本与显存
configs/               每矩阵正序/反序配置
raw/                   每独立进程的逐轮 CSV、布局/构建日志
status/                每矩阵的输入哈希、命令、退出码、耗时与干扰状态
provenance/            对应冻结二进制、源码快照与分析脚本
completed.json         批次结束记录；必须同时检查 failures
summary.json           成功数、失败项、有效候选对数量
timings.csv            每候选两进程中位时间及几何平均
factors.csv            主实验逐矩阵配对因素收益
factor-summary.json    主实验各因素的覆盖数、几何平均及正负收益数
row-order-factors.csv  行序实验的槽数、密度变化及配对时间比
coverage-audit.json    原始样本数、manifest 候选覆盖、退出状态与哈希核验
```

最后四个结果文件按实验类型生成。`PLAN_LAYOUT` 的字段依次为 backend、原矩阵行/列数、nnz、实际 tile 编号、balanced、grouped、compressed、压缩列元数据字节数。普通格式该字节数字段为 0，不能据此推导其总存储为零。重复 backend 按配置出现，顺序对应本进程的 config 文件；`atbsr_control` 在 CSV 中记录实际解析出的配置。

配置一般为 `backend,tile,balance,parts,threads,group,threshold`。额外消融约定只属于基准程序：

- `atbsr_historical`：构建期间将 prefetch 设为 0，其他参数恢复公共设置。
- `atbsr_tuning`：最后四项解释为 `cap,warps,prefetch,cta_merge`，不改库的默认选项。
- `mixed_tile`：parts 项解释为 mixed cap（0 或 8），不进行初始化调优。
- `atbsr_clear`：普通模式计算前显式清零 y，比较省去清零的效果。
- `atbsr_i32`：固定 shape/模式，将声明列维度扩到 32767，以触发 int32 存储；未增加实际非零元或有效 gather。该实验所有候选的 X 都补到至少 32767 个元素，额外元素为零。
- `atbsr_rows_sorted/original`：同一个独立 host packer 开/关稳定行长排序，执行同一个生产 `AtSpmvProduct` kernel。它不是 CSRToAtBSRProduct GPU 转换速度的对照。

主实验固定普通参数 depth=2、4 warp/CTA、balanced cap=8、CTA merge=0；个别候选只覆盖标明的变量。depth=0 和 depth>0 选择不同生产内核实现，因此预取对照包含这一执行路径变化，不应拆解成单条预取指令的贡献。grouped 自带两路累加，不受 depth=4 候选影响。

`shape_oracle_vs_32` 是固定 32×1 普通时间 / 六种普通中的最快时间；`balance_oracle_vs_ordinary` 是六种普通最快 / 十二种普通+balanced 最快。二者是事后覆盖对照，天然不小于 1，不代表在线选型。`mixed_oracle_vs_fixed12` 两边也是事后最优；本轮没有给固定所有 shape 枚举 cap 4/16，不能代替已有 80 矩阵公平 cap 对照。

运行或重建统计（设备需计算空闲；物理卡号与 metadata 的 UUID 必须一致）：

```bash
chmod +x results/model/optimization-ablation/h100-20261111/provenance/spmv_hybrid_validation
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/analysis/benchmark_optimization_ablation.py \
  --output results/model/optimization-ablation/h100-20261111 \
  --exe results/model/optimization-ablation/h100-20261111/provenance/spmv_hybrid_validation
python3 scripts/analysis/analyze_optimization_ablation.py results/model/optimization-ablation/h100-20261111
python3 scripts/analysis/analyze_optimization_ablation.py results/model/optimization-ablation/h100-row-order-20261111 --row-order
```

已有成功项会跳过；不修改输入/驱动/二进制身份。修改实现后应另建实验目录，不覆盖成功数据。使用 `--analyze-only` 只核验身份并生成基础统计；因素分析由第二个脚本生成。新启动可放入 `tmux`，当前两批均已结束，无需等待。两次手动冒烟失败发生在正式实验前，修正了测试参数重载和 X 扩展后的区间验证，正式 9+4 矩阵全部成功；冒烟不计入任何统计。
