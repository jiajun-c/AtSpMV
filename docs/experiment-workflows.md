# 历史实验工作流与工具

以下保留原 README 中的绘图、模型、DSM 与历史数据入口，命令均从仓库根目录执行。阶段性指标以相应冻结实验及专题报告为准；当前构建入口见 [README](../README.md)，文档导航见 [文档索引](README.md)。

## 绘图

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
MPLBACKEND=Agg python scripts/analysis/draw_density_perf.py --output artifacts/analysis_results
# 交互显示：添加 --show；加载指定字体：添加 --font /path/to/font.ttf。
```

该入口保留原始实验数值及绘图布局，生成 SVG 和 PDF。默认使用已安装的 Times New Roman，否则回退到 DejaVu Serif。导入模块不会创建图形或写入文件。分析脚本通过自身位置定位 `results/csv/` 和 `results/figures/`，可从任意工作目录执行；显式 `--output` 仍相对于当前工作目录。历史脚本的外部数据与字体依赖见下文。

## 开发与验证

CMake 将核心实现 (`atspmv_core`) 与对照实现 (`atspmv_baselines`) 分开，`atspmv` 保留原有 CMake 链接入口；Plan 和普通测试只依赖核心库。新增算法需更新源文件列表；新增实验需注册可执行目标。

```bash
python -m compileall -q scripts/analysis
bash -n scripts/run_benchmarks.sh
MPLBACKEND=Agg python scripts/analysis/draw_density_perf.py --output /tmp/atspmv-check/analysis
```

CUDA 改动需要在目标 GPU 上编译并与 CPU/cuSPARSE 对比，覆盖空行、不等长行、不同块形状和索引宽度。当前项目仍是实验实现，不能仅凭成功编译判断数值正确，具体限制见算法说明。

## 历史分析与数据依赖

- 原根目录的 Python 脚本已迁入 `scripts/analysis/`，文件名保留；`bench/test.py` 改名为 `check_diagonal_dominance.py`，用于检查指定矩阵的对角占优性。
- `draw.py` 需要 `results/csv/fullres.csv`；Notebook 需要 `results/csv/fullresbsr.csv` 或 `matrix_row_len.csv`。这些输入原本未随仓库提供，需自行补齐。Notebook 可从仓库根目录或 `notebooks/` 启动。
- 多数历史绘图脚本仍使用原工作区 `tcspmv/times.ttf`、`timeBold.ttf` 字体路径；移植到其他机器时需修改这些路径。推荐使用支持 `--font` 和字体回退的 `scripts/analysis/draw_density_perf.py`。
- `export_excel.py` 从仓库同级的 `tcspmv/fullres.csv`、`tcspmvh100/fullres_h100.csv` 读取历史实验数据；可用 `ATSPMV_WORKSPACE` 指定它们所在的父目录。导出表格写入 `results/tables/`。
- `check_diagonal_dominance.py` 保留历史外部矩阵路径，使用前按目标矩阵调整。
- PageRank 数据包说明见 [results/pagerank/数据说明.md](../results/pagerank/数据说明.md)，其脚本将结果写回包内。
- 原 `test.sh` 已迁入 `scripts/run_benchmarks.sh`，默认使用仓库的 `build/spmv_double`，日志写入 `artifacts/runs/`。

历史绘图脚本会覆盖对应的已有图表；需要保留不同实验版本时，请先备份或调整输出文件名。`draw_abaltion.py` 与 `draw_abaltion_int16.py` 仍共享 `bar.*` 输出名。

## Tile 时间模型

AtSpMV 按预测运行时间选择六种块形状和普通/均衡调度模式，包含 SM 数、occupancy、缓存
容量和长尾/原子开销。RTX 5880 Ada 已内置预热与冷缓存校准参数；其他硬件使用未校准默认值。
40 个独立留出矩阵、每种状态 480 个候选：预热 MAPE 14.15%、P90 34.16%；冷缓存 MAPE
15.40%、P90 30.23%。独立复测也通过 MAPE≤30%、P90≤50% 门槛。
详细公式、最差案例及重现方法见 [SM 时间模型](performance-model.md)。

```bash
ctest --test-dir build --output-on-failure
CUDA_VISIBLE_DEVICES=0 ./build/spmv_regression
mkdir -p artifacts
CUDA_VISIBLE_DEVICES=0 ./build/tile_model_bench --synthetic > artifacts/tile-model.csv
```

`ATSPMV_MODEL_TRACE=1` 输出所有候选的预测分项。`tile_model_bench` 用 CPU 参考结果校验
每个候选，并输出预测与 CUDA Graph/event 实测时间；运行前请选空闲 GPU。
`ATSPMV_BENCH_CACHE=cold` 可测试冷缓存；应用中用 `ATSPMV_MODEL_CACHE=cold` 选择相应模型。


可用 `ATSPMV_TUNE_CAP=4|8|16|32`、`ATSPMV_TUNE_WARPS=2|4|8`、
`ATSPMV_TUNE_PREFETCH=0|1|2|4|8` 与 `ATSPMV_TUNE_CTA_REDUCE=0|1` 进行内核对照实验。
非默认配置尚未校准，不能套用上述误差指标。默认普通模式 `beta=0` 已去掉冗余清零启动。
配置语义、离线扫描脚本和实测结果见 [可配置内核实验](performance-model.md#可配置内核实验)。


另有默认关闭的 SM90 实验：`-DATSPMV_BUILD_DSM_EXPERIMENT=ON` 构建
`atspmv_dsm` 与 `dsm_spmv_test`，将完整 x 分片存入 cluster 的分布式共享内存。
H100 正确性验证已通过；完整 x DSM 的单次 SpMV 测试未发现收益。
新增 `dsm_pagerank_bench` 与持久化 PageRank 接口：在 netscience、NotreDame_yeast
等真实图上，跨迭代复用 DSM 获得了可重复收益，但并非所有图都适用。
对照结果、复现方式和限制见 [持久化 PageRank 实验](performance-model.md#持久化-pagerank-的-dsm-实验)。
完整 x 的构建、容量限制及接口见
[DSM 实验](performance-model.md#完整-x-的分布式共享内存实验sm90)。


持久化 GEMV 调度对照见 [GEMV 实验](performance-model.md#gemv隔离持久化行分配与负载不均衡)：
显式比较一行一 CTA、循环分行、连续分行，并导出每个 CTA 的行集合。
该基准不使用 DSM，用于区分负载均衡与网格并行度的影响；启用上述 SM90 实验构建后，
目标为 `gemv_persistent_bench`。

单次 SpMV 的长行规约另有 `dsm_long_row_bench` 与 `launchDsmLongRows`：参考 Quack，
使用异步 DSM store 和 transaction barrier 交换 FP64 部分和，支持在同一 kernel 处理短行。
它不缓存完整 X；适用范围、对照结果和验证记录见
[长行 DSM 规约](performance-model.md#单次-spmv-的长行-dsm-规约)。

统一调用入口为 `atspmv_plan` 库的 `makeSpmvPlan` / `executeSpmv`。
默认沿用 AtBSR 模型；设置 `SpmvPlanOptions::tune=true` 可在初始化时实测选择 AtBSR、
CSR vector、cuSPARSE、CSR 两阶段及可用的 DSM 混合后端。也可显式开启 `model_select` 和
`vector_model_select`，在已校准的 H100 PCIe 范围内使用统一模型；额外开启
`model_fallback_tune` 时，模型范围外改为初始化实测。完整实测可选用
`retry_shared_storage_on_oom`，在识别到 GPU 分配失败后用共享打包重试一次。模型和重试
均需显式启用，初始化成本与适用范围见下方文档。Plan 拥有矩阵副本和工作区，支持 CUDA Graph 执行。
详见 [统一 Plan](performance-model.md#统一-spmv-plan-与初始化实测选型)。

### 统计最长行比例

下面的命令统计矩阵清单中每个 Matrix Market 坐标矩阵的最长行。比例为最长行的 CSR 条目数除以矩阵总 CSR 条目数；对称、反对称和 Hermitian 矩阵先展开，重复坐标和显式零均保留，与项目读取器的计数方式一致。行号从 1 开始，最长行并列时返回第一行；空矩阵的比例为空。缺失或格式有误的矩阵会在 `status` 列中标出。

```bash
g++ -O3 -std=c++17 tools/longest_row_ratio.cpp -o /tmp/longest_row_ratio
/tmp/longest_row_ratio datasets/matrix_list.txt /path/to/suitsparse > longest_row_ratio.csv
```

数据目录使用 `DATA_ROOT/矩阵名/矩阵名.mtx` 布局。`ratio` 是 0 到 1 之间的比例，乘以 100 即百分比。
