# AtSpMV

**Model-Guided Adaptive Tiling and Load Balancing for SpMV on GPUs**

基于 CUDA 的 FP64 稀疏矩阵向量乘研究实现。AtSpMV 按行长分组，在六种包含 32 个槽位的 tile shape 中选择线程映射，并结合时间模型与长行任务拆分适应不同稀疏结构。核心计算使用普通 FP64 乘加和 warp 归约。

[论文 DOI](https://doi.org/10.1007/978-3-032-35248-4_35) · [论文 PDF](docs/atspmv.pdf) · [引用](CITATION.bib) · [文档导航](docs/README.md) · [代码架构](docs/architecture.md)

## 论文与仓库范围

**AtSpMV: Model-Guided Adaptive Tiling and Load Balancing for SpMV on GPUs**

Jiajun Cheng, Junshi Chen, Qi Li, Longsheng Song, Yang Zhao, Jun Shi, Xiaoyu Hao, Hong An.

*Euro-Par 2026: Parallel Processing*, LNCS 16781, pp. 509–522.

DOI: [10.1007/978-3-032-35248-4_35](https://doi.org/10.1007/978-3-032-35248-4_35).

仓库同时包含论文相关实现和后续研究扩展，当前工作树并非冻结的论文 artifact：

| 层次 | 内容 | 入口 |
|---|---|---|
| AtSpMV 核心 | GPU 行排序与打包、六种 shape、普通/均衡模式、时间模型 | [算法](docs/algorithm.md)、[时间模型](docs/performance-model.md) |
| 统一 Plan | 资源管理、后端选择、可选初始化实测、固定结构数值更新 | [接口与选型](docs/unified-mixed-plan.md) |
| 扩展实验 | 分组列索引、mixed-tile、DSM 长行规约、SpTRSV/SpMM 探索 | [优化消融](docs/optimization-report.md)、[文档索引](docs/README.md) |
| 应用 | PageRank、Jacobi、可选 MFEM 热传导 | [apps/](apps/)、[MFEM](apps/mfem/README.md) |

默认 Plan 沿用 AtBSR 的模型选择；mixed、分组列索引、统一模型精调和 DSM 等功能有独立开关，不能把所有扩展的收益当作论文原方案的收益。

## 快速开始

需要 CMake 3.22+、支持 C++17 的 CUDA/宿主编译器、CUDA Toolkit（cuSPARSE、cuBLAS）和 OpenMP。普通 AtSpMV 路径不要求 Tensor Core；默认构建 SM80，按实际设备修改架构。DSM 另需 CUDA 12+ 与支持 cluster 的 SM90 设备。

```bash
# H100 示例；A100 使用 80。
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build --target spmv_plan_bench --parallel

# 读取 Matrix Market、构建默认 Plan，并检查 CPU 参考结果。
CUDA_VISIBLE_DEVICES=0 ./build/spmv_plan_bench datasets/matrices/test.mtx
```

首次配置会下载固定提交的 `fast_matrix_market`。离线构建可追加：

```bash
-DFETCHCONTENT_SOURCE_DIR_FAST_MATRIX_MARKET=/path/to/fast_matrix_market
```

也可使用预设减少参数和不必要的构建目标：

```bash
cmake --preset core                 # 仅核心 SpMV 库
cmake --build --preset core --parallel
# 将 core 换为 benchmarks：常用基准与检查；换为 hopper：SM90/DSM 实验。
```

预设的构建目录分别为 `build/core`、`build/benchmarks`、`build/hopper`，不覆盖现有
`build/` 配置。机器专用编译器/依赖路径放在命令行或被忽略的 `CMakeUserPresets.json`。

需要指定 CUDA 或宿主编译器时，在首次配置时设置 `CMAKE_CUDA_COMPILER`、`CMAKE_CUDA_HOST_COMPILER` 与 `CMAKE_CXX_COMPILER`；更换编译器使用新构建目录。当前已验证环境包括 CUDA 12.6、GCC 11、H100 PCIe，详见各实验的冻结配置。

### 其他运行方式

```bash
# 初始化时实测已有候选，代价计入 Plan 构建。
./build/spmv_plan_bench /path/to/matrix.mtx --tune

# 模型候选、CSR vector 和 mixed-tile 的有界实测。
./build/spmv_plan_bench /path/to/matrix.mtx --model-refine-mixed

# 论文相关传统对照入口；第三方后端的计时/正确性限制见架构文档。
cmake --build build --target spmv_double --parallel
./build/spmv_double datasets/matrices/test.mtx

# PageRank 应用。
cmake --build build --target pagerank_atspmv --parallel
./build/pagerank_atspmv /path/to/graph.mtx
```

`spmv_plan_bench --model-refine-mixed` 是该命令行入口的配置，不等同于下方 100 矩阵实验额外启用 vector-v2 的配置。复现该结果应使用冻结清单及对应实验驱动。

## 集成到应用

链接 `atspmv_plan`，使用 [include/spmvPlan.h](include/spmvPlan.h)：

```cpp
atspmv::SpmvPlanOptions options;
auto plan = atspmv::makeSpmvPlan(device_csr, options, stream);
atspmv::executeSpmv(plan, device_x, device_y, stream);
```

输入 CSR、X、Y 位于设备端；当前入口使用 FP64 值和 int32 CSR 索引，计算 `y=A*x`。Plan 拥有矩阵副本和工作区，必须活到提交的操作和引用它的 Graph 结束。数值更新、并发和所有权要求见 [Plan 文档](docs/unified-mixed-plan.md)。

## 构建选项与验证

| 选项 | 默认值 | 作用 |
|---|---|---|
| `ATSPMV_BUILD_BENCHMARKS` | ON | 基准与分析可执行程序 |
| `ATSPMV_BUILD_APPS` | 首次跟随 benchmarks | PageRank、Jacobi |
| `BUILD_TESTING` | ON | 构建测试目标并注册 CTest |
| `ATSPMV_ENABLE_GPU_TESTS` | OFF | 在 CTest 中启用 GPU 测试 |
| `ATSPMV_BUILD_DSM_EXPERIMENT` | OFF | SM90 DSM 扩展 |
| `ATSPMV_BUILD_MFEM_APPS` | OFF | 外部 CUDA/FP64/串行 MFEM v4.8 应用 |

```bash
cmake --build build --target tile_model_test spmv_value_update_test --parallel
ctest --test-dir build -R '^tile_model$' --output-on-failure
CUDA_VISIBLE_DEVICES=0 ./build/spmv_value_update_test
```

GPU 测试和性能实验应选空闲设备。大范围测试可能包含实验路径，检查逐项退出状态；成功编译不代表数值正确。第三方包装、alpha/beta 和历史入口的已知限制见 [架构](docs/architecture.md) 与 [算法](docs/algorithm.md)。

## 最新扩展实验：100 矩阵对照

以下是后续统一框架的分层样本结果，**不是论文实验结果**。同一 H100 上 100/100 矩阵通过检查，相对每轮 cuSPARSE DEFAULT/ALG1/ALG2 中实测最快者：

| 方案 | 几何加速比 |
|---|---:|
| 原 AtBSR | 0.901× |
| 分组 AtBSR | 0.899× |
| 统一框架：vector-v2 + 有界实测 + mixed | 1.287× |
| 上述框架允许分组 | **1.295×** |

结果为预热 Graph 稳态时间，排除转换和选型成本；样本有意覆盖规模和长行结构，不能推广为全库平均。主要收益来自后端选择。完整方法、退化案例、初始化成本和复现入口见 [100 矩阵报告](docs/spmv-sample100.md)。

## 目录与阅读顺序

```text
include/           公共接口、tile/统一模型及 detail 内部实现
src/               核心转换、kernel 和 Plan
  baselines/       AMG-T、DASP、cuSPARSE 对照包装
  selection/       主机选型逻辑及校准参数
apps/              PageRank、Jacobi、可选 MFEM
bench/             基准入口；detail/ 为实验专用辅助
cmake/             基准、应用、测试及 DSM 的构建注册
scripts/analysis/  抽样、跑批、分析与审计
models/            校准数据与模型元信息
tests/             CPU/GPU 正确性与分析回归
datasets/          随附样例、矩阵清单及历史数据包
results/           原始测量、冻结输入与版本快照、汇总
docs/              算法、接口、实验报告和论文
```

从 [代码架构](docs/architecture.md) → [算法](docs/algorithm.md) → [统一 Plan](docs/unified-mixed-plan.md) 开始阅读。更多专题见 [docs/README.md](docs/README.md)。绘图、批处理、字体和历史数据依赖集中在 [历史工作流](docs/experiment-workflows.md)。本地生成内容放在 `build/`、`artifacts/` 或 `.venv/`。

新增 C++/CUDA 代码遵循根目录 `.clang-format`。生产代码放在 `src/` 与 `include/`，
实验标签和参考验证放在 `bench/detail/`；冻结实验的二进制、源快照和原始结果不随重构覆盖。

## 源码发布与实验归档

GitHub 源码版保留实验脚本、源快照、CSV/JSON、文本日志及图表；生成的可执行文件、
内存/模型二进制、派生实验矩阵和指向本机数据集的符号链接不随源码上传。
文本结果可以重新汇总；重新跑批需按清单准备输入并重新构建。
需要核验原二进制哈希或完整温度场时，须使用对应完整实验归档，不能仅靠源码版完成该项审计。

## 引用

请使用 [CITATION.bib](CITATION.bib) 或 [CITATION.cff](CITATION.cff)。出版社页面将首次在线发表日期标为 2026-08-15，但推荐引用年份为 2027；本仓库引用文件遵循[出版社推荐格式](https://link.springer.com/chapter/10.1007/978-3-032-35248-4_35#citeas)。
