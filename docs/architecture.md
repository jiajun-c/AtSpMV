# 代码架构与阅读顺序

这个仓库同时包含可复用的 SpMV 实现、Plan/选型代码和研究实验。建议从接口和一次执行路径开始，再按需要查看具体 kernel；历史实验及生成数据不参与默认调用路径。

## 构建入口

根 `CMakeLists.txt` 定义核心库及其依赖，随后按原有顺序加载四个注册模块：

| 模块 | 职责 |
|---|---|
| [Benchmarks.cmake](../cmake/Benchmarks.cmake) | 基准程序及 `ATSPMV_BUILD_BENCHMARKS` |
| [Applications.cmake](../cmake/Applications.cmake) | PageRank、Jacobi 和可选 MFEM |
| [Tests.cmake](../cmake/Tests.cmake) | 测试目标、CTest 注册和 GPU 测试开关 |
| [DsmExperiment.cmake](../cmake/DsmExperiment.cmake) | 可选 SM90 库、DSM 基准与测试 |

这些文件共享根目录作用域；拆分保留原目标名、依赖和选项默认值。新增实验在相应模块注册，
不将实验专用配置加入公共 Plan 接口。构建命令见 [README](../README.md)。

## 实验代码的边界

`bench/spmv_hybrid_validation.cu` 负责读取候选、建立 Plan、校验结果和配对计时。
实验标签及临时参数不进入生产 API，公共逻辑按职责放在 `bench/detail/`：

| 文件 | 职责 |
|---|---|
| `spmvBenchmarkOptions.h` | 将实验标签映射到公开的 `SpmvPlanOptions` |
| `spmvBenchmarkSupport.cuh` | CUDA 检查、上传、候选记录和临时环境变量的自动恢复 |
| `spmvRowOrder.cuh` | 仅供行排序消融使用的 host packer，不参与生产转换 |
| `spmvBenchmarkValidation.h` | 两组固定 X 与 CPU 参考值，直接/Graph 共用误差检查 |

新测量优先复用 `scripts/analysis/benchmark_optimization_ablation.py`：输入为冻结 manifest，
输出原始进程日志与 CSV；汇总脚本只读取这些输出。自动选型标签按方案配对，实际选中的
backend/config 在各进程原始 CSV 中保留，允许两个顺序选中不同配置。
100 矩阵抽样、对照与实验范围见 [100 矩阵报告](spmv-sample100.md)。

## 一次 Plan 调用

```text
CSR(device) + SpmvPlanOptions
  └─ makeSpmvPlan                         src/spmvPlan.cu
     ├─ 验证 CSR、复制输入和获取设备能力
     ├─ 创建候选和所需的矩阵布局
     │  ├─ 单一 AtBSR                    src/atspmvProduct.cu
     │  ├─ 混合 tile                    include/detail/mixedTile.cuh
     │  └─ DSM 长行及短行组合            src/spmvPlanDsm.cu
     ├─ 可选模型筛选                     src/selection/spmvPlanModel.cpp
     ├─ 可选预热 Graph 实测并选择候选
     └─ 保留赢家、释放未选中的候选

executeSpmv(plan, x, y, stream)
  └─ 按赢家 backend 分派到相应 kernel；不在此阶段重新选型或分配
```

外部接口与所有权约束在 [spmvPlan.h](../include/spmvPlan.h)：Plan move-only，源 CSR 可在构建后释放；Plan 必须活到已提交工作和引用它的 Graph 结束，每个并发 stream 使用独立 Plan。

两套 AtBSR 转换入口共用 `include/detail/rowLengths.cuh` 的行长计算。
Product 的 legacy/tuned kernel 由一个选择函数提供给 occupancy 预测与执行分派，
不再为两者维护重复的 shape/索引宽度判断。内部无调用、未公开声明的 `SpmvProd` 已移除；
主机 `Spmv` 与设备 `CSRToAtBSRProduct` / `AtSpmvProduct` 接口保持不变。

## 主要模块

| 模块 | 位置 | 职责 |
|---|---|---|
| Plan 接口 | `include/spmvPlan.h` | backend、可选项、结果信息、Plan 生命周期 |
| Plan 构建与调度 | `src/spmvPlan.cu` | 候选构建、模型/实测选型、内存所有权、执行分派 |
| 实测候选决策 | `include/detail/planTimingSelection.h` | 独立的实测时间比较、混合方案收益门槛与资格过滤 |
| Plan 资源与候选 | `include/detail/planStorage.cuh` | 候选、缓冲区、打包布局与 cuSPARSE 描述符的所有权 |
| Plan 通用 kernel | `include/detail/spmvPlanKernels.cuh` | CSR vector 与两阶段规约 kernel |
| 单一 AtBSR | `include/atspmvprod.h`、`src/atspmvProduct.cu` | CSR 打包、单一 shape 的普通与均衡 SpMV |
| 单一 tile 模型 | `include/tileModel.h`、`include/tileSelector.cuh` | shape 的统计和时间预测 |
| 混合 tile | `include/detail/mixedTile.cuh` | 排序、动态规划、打包、任务切分与执行 |
| DSM backend | `include/detail/spmvPlanDsm.h`、`src/spmvPlanDsm.cu`、`include/dsmLongRow.cuh` | cluster 长行规约和 Plan 适配；需要 SM90 构建与设备支持 |
| 统一选型模型 | `src/selection/spmvPlanModel.cpp`、`include/detail/planModelSelection.h`、`include/unifiedSpmvModel.h`、`include/hybridModelFeatures.h` | 不拥有 GPU 资源的模型决策、特征与预测参数 |
| CSR 读取 | `include/csrReader.h`、`src/csrReader.cpp` | Matrix Market 到 host CSR；当前 FP64 实数 |
| 第三方对照 | `src/baselines/amgT.cu`、`src/baselines/dasp.cu`、`src/baselines/cusparseCSR.cu` | 基准所用的 AMG-T、DASP 与 cuSPARSE 包装，不参与 Plan 后端选择；AMG-T 包装有累加行为，见下方限制 |

`Auto` 默认使用单一 AtBSR 的现有模型；`model_select=true` 可启用有范围约束的 H100 统一模型。`tune=true` 使用候选实测；`enable_mixed_tile=true` 且 `tune=true` 时加入五个混合 tile 候选。`model_refine=true` 可选择对模型候选和 CSR vector 做有界实测。混合 tile **没有接入预测模型**，也没有实现 DSM 长行与混合 tile 短行的组合。详情见 [统一 Plan 使用](unified-mixed-plan.md)。

## 构建、测试和实验

- `CMakeLists.txt`：`atspmv_core` 是基础库，`atspmv_baselines` 是独立对照库，`atspmv` 是兼容 CMake 链接入口；`atspmv_plan` 只依赖基础库；可选 `ATSPMV_BUILD_DSM_EXPERIMENT` 增加 SM90 DSM backend。`bench/` 的可执行文件依赖这些库，不是新的生产后端。
- `tools/plan_model_replay.cpp`：在 CPU 上重放模型决策；`scripts/analysis/audit_plan_model_replay.py` 可与冻结的 GPU Plan 选项对照。
- `tests/spmv_plan_test.cu`：Plan 的直接执行、Graph、生命周期和数值检查；`tests/spmv_regression.cu` 覆盖基础 SpMV；`tests/baseline_regression.cpp` 检查对照接口及其已知历史行为。
- `apps/`：PageRank（cuSPARSE 和 AtSpMV 两个入口）及 Jacobi 应用，使用独立的 `ATSPMV_BUILD_APPS` 构建选项。
- `bench/spmv_plan_bench.cu`：单矩阵查看候选、初始化时间及正确性；`bench/spmv_hybrid_validation.cu`：配对测量方案选择；`bench/mixed_tile_bench.cu`：混合布局与单一 shape 的专门对照。
- `scripts/analysis/`：研究脚本和历史绘图。与 Plan 选型有关的入口为 `benchmark_plan_selection.py` 和 `analyze_plan_selection.py`；结果、原始日志和冻结快照位于 `results/model/`，并不作为运行时模型直接加载。
- `docs/performance-model.md`、`docs/dsmem-background.md` 和 `docs/mixed-tile-profile.md`：算法与实验背景；较长的历史结果不属于 API 文档。

GPU 测量须注明硬件、候选范围、初始化是否计时和缓存状态。实测调优得到的是它实际枚举、构建并测量的候选中的选择，且可能有收益门槛和计时噪声；不能称为全算法空间的全局最优。

模型相对实测候选的最新配对验证见 [H100 选型验证](selection-model-validation.md)。

## 现有对照接口的限制

`amgT_spmv_fp64` 是历史基准包装：所用 kernel 向输出累加，包装执行 10 次预热和 `repeat` 次计时而不重置输出。初始 `y` 为零时，所测案例最终返回 `(10 + repeat) × A*x`，而不是单次 `A*x`。本次目录整理保留了这一行为；它不应作为数值正确的 SpMV 接口或参考结果，历史计时也不能直接与包含输出清零的正确实现比较。`baseline_regression` 对两个本地验证案例明确检查这一历史行为，同时检查 DASP 的行号映射及 cuSPARSE 的输出。
