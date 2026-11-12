# 文档导航

构建和使用从 [项目 README](../README.md) 开始。本文档按问题组织入口，历史数据保持原位置，避免移动冻结实验后破坏引用。

## 核心代码与接口

| 问题 | 文档 |
|---|---|
| 代码从哪里读，模块如何分工？ | [架构与调用路径](architecture.md) |
| 行排序、六种 shape、均衡模式如何工作？ | [算法流程](algorithm.md) |
| 如何创建、执行、更新和调优 Plan？ | [统一 Plan](unified-mixed-plan.md) |
| 时间模型预测什么，如何校准？ | [SM 性能模型](performance-model.md) |
| 如何绘图、批处理和使用历史脚本？ | [实验工作流与依赖](experiment-workflows.md) |

## 当前性能与消融

| 问题 | 文档 |
|---|---|
| 当前整体方案相比 cuSPARSE 如何？ | [100 矩阵对照](spmv-sample100.md) |
| 当前使用了哪些优化，分别有什么效果？ | [优化与消融总览](optimization-report.md) |
| 分组索引能否覆盖六种 shape？ | [六种 shape 扩展](grouped-columns-six-shapes.md) |
| 原 32×1 分组布局的历史矩阵结果？ | [24 矩阵复测](grouped-columns-prior-matrices.md) |
| 模型是否能选到较快后端？ | [选型验证](selection-model-validation.md)、[历史全库快照](selection-model-current-snapshot.md) |

100 矩阵、24 矩阵及历史全库快照的候选集合和样本不同，不能直接相减判断版本变化；后续扩展结果也不等于论文原始实验。

## 研究扩展

- **Mixed-tile：** [密度](mixed-tile-density.md)、[配对性能](mixed-tile-experiment.md)、[profile](mixed-tile-profile.md)、[选择条件](mixed-tile-selection-analysis.md)。
- **DSM：** [背景与时延](dsmem-background.md)、[长短行融合](dsmem-hybrid.md)、[扩展矩阵结果](dsmem-full-sweep.md)。
- **MFEM：** [应用流程](mfem-application.md)、[L2/旧压缩消融](mfem-column-ablation.md)、[原生分组布局](mfem-native-atbsr.md)、[构建](../apps/mfem/README.md)。
- **其他算子：** [SpTRSV](sptrsv-experiment.md)、[SpMM 指令路径对照](spmm-core-comparison.md)、[极稀疏块](spmm-exact-one-block.md)、[扩展方向](sparse-extension-research.md)。

## 论文与实验包

- [论文 DOI](https://doi.org/10.1007/978-3-032-35248-4_35)、[仓库 PDF](atspmv.pdf)、[BibTeX](../CITATION.bib)。
- [100 矩阵实验结构](../results/model/spmv-sample100/README.md)。
- [因子消融实验结构](../results/model/optimization-ablation/README.md)。
- [历史 PageRank 数据包](../results/pagerank/数据说明.md)。

每项性能结论以专题报告中的设备、候选范围、初始化是否计入、成功/失败项及对应冻结数据为准。
