# 统一 Plan 选型的 H100 配对验证

## 问题与范围

问题是：当前默认 H100 统一模型选出的方案，是否与实际测到的较快方案一致。实验在一张空闲 NVIDIA H100 PCIe 上，从已有 389 矩阵清单用固定随机种子 20261028 抽取 40 张；40 张均通过正确性检查。这个子样本可能包含相似矩阵族，且不能仅凭它证明训练集外泛化或全库表现。

每个矩阵在同一测试进程中分别构建并独立测量：

- `auto_model`：`model_select=true` 的现有 H100 模型；未启用实验性的 vector 模型，也未包括 mixed-tile 预测。
- `auto_tuned`：已有候选集的完整初始化实测选型。
- `auto_tuned_mixed`：同一完整调优集增加五种 mixed-tile 候选；默认 5% 收益门槛仍然生效。
- 默认单一 AtBSR 和预处理后的 cuSPARSE CSR_ALG1：作为固定基线。

测量包含两种相反的 Plan 构建顺序，每个配置在预热 Graph 内执行 8 次，3 轮预热、9 轮随机交错计时并取中位数；每种顺序的比值取几何平均。每个 Plan 用两组输入验证直接结果，Graph 用第二组输入验证。时间不包括输入读取、构建和初始化选型。采样式 GPU 干扰监测不能排除所有干扰；事件节点、不同的独立 Plan 分配与 Graph 也可能影响微秒级结果。

这里的“实测最好”是五个**已选出并独立复测的 Plan**中每种顺序最快者，不是全部可能配置的全局最优。`auto_tuned` 和 `auto_tuned_mixed` 的候选是初始化时测出的，选择本身受到测量噪声和 5% 门槛影响；只有选中者被独立复测。因而五个 Plan 中最快值是一个有用的对照，也仍可能慢于未复测的某个候选。

## 结果

| 指标 | 结果 |
|---|---:|
| 成功案例 | 40 / 40 |
| 模型时间 / 五个实测 Plan 中最好时间，几何平均 | 1.158× |
| 上述比值的中位数 | 1.142× |
| 模型在 5% 范围内的矩阵 | 8 / 40 |
| 模型慢超过 10% 的矩阵 | 29 / 40 |
| 最大模型时间 / 五 Plan 最好时间 | 1.454× |
| `auto_tuned_mixed` 相对 `auto_model` 的加速比几何平均 | 1.146× |
| `auto_tuned_mixed` 相对 `auto_tuned` 的加速比几何平均 | 0.998× |

在第一种记录顺序下，模型选到 AtBSR 30 张、两阶段 CSR 5 张、CSR DSM 4 张、DSM/AtBSR 混合 1 张；含 mixed-tile 的实测选型在两个顺序都选 CSR vector 的案例有 28 张、都选 AtBSR 的有 7 张、都选 mixed-tile 的有 1 张，另有少量其他或顺序间不同的选项。模型的候选空间没有启用 vector 模型，而 tuned 实测包含 CSR vector，这是观察到差距的一个明确候选空间差异，不能仅归因于模型回归精度。

例如 `n4c6-b13` 的模型配置是 AtBSR，实测选型是 CSR vector width 16；模型时间相对五个 Plan 中最好时间为 1.454×。其他差距较大的矩阵也存在模型 AtBSR 对照实测 CSR vector 的情况；这表明启用、校准和独立验证 CSR vector 模型值得优先研究。MixedTile 本身在这个小样本中仅被两个顺序均选中 1 次，不能据此断言在全库中普遍有用或无用。

本结果说明当前默认模型不能保证选择这些实测候选中最快的方案。它不能说明模型一定比完整调优的每个候选差，也不能说明大工作集、冷缓存、其他 GPU 或整个矩阵库的结论。开启模型侧 vector 扩展的对照、按矩阵族分组的独立验证以及同一候选配置的独立复测是下一步。

## 复查

冻结结果及输入、可执行文件和脚本在 [h100-paired40](../results/model/selection-model/h100-paired40/)；逐矩阵对照见 [oracle-analysis.csv](../results/model/selection-model/h100-paired40/oracle-analysis.csv)。重新计算统计：

```bash
python3 scripts/analysis/analyze_plan_selection.py results/model/selection-model/h100-paired40
python3 scripts/analysis/analyze_selection_oracle.py results/model/selection-model/h100-paired40
```

## CSR vector 模型的同批受控对照

继续使用同一份 40 张清单，在另一轮 H100 测试中增加 `auto_vector_model`。先构造默认 AtBSR 作为预热锚点，再交换其余 Plan 的构建顺序；原模型和 vector 模型选中相同配置时共用 Plan 和 Graph，事件放在 Graph 内。仍比较完整调优和包含 mixed-tile 的完整调优；这些 Plan 没有共用 Graph，独立 Graph 与分配差异仍可能影响结果。全部 40 张通过正确性检查。这是已查看过的样本上的复测，不是新的盲测。

| 指标 | 原模型 | 启用 vector 模型 |
|---|---:|---:|
| 相对六个已测 Plan 最快时间的比值几何平均 | 1.172× | 1.049× |
| 距实测最快不超过 5% | 8 / 40 | 30 / 40 |
| 慢于实测最快超过 10% | 30 / 40 | 6 / 40 |

vector 模型相对原模型的加速比几何平均为 1.117×，但 `bibd_17_4` 上仅为 0.757×，原因是预测选择 vector width 32，而该测试中的较快实测方案是 AtBSR。单一样本不能证明对全库收益，也说明直接启用 vector 模型无法避免较大误选。[逐矩阵对照](../results/model/selection-model/h100-vector-paired40/oracle-analysis.csv)，[汇总](../results/model/selection-model/h100-vector-paired40/oracle-analysis.json)。

统计 Plan 构造日志得到原模型、vector 模型、完整调优和含 mixed-tile 完整调优的初始化耗时中位数分别约为 4.58、5.19、208 和 222 ms；这些是该轮按 Plan 分别统计的样本中位数，不应直接用其相减作为逐矩阵额外成本。首次准备由默认 AtBSR 锚点承担，构造顺序、内存分配和系统负载也会影响该数值。初始化未包含在上表运行时间内。

模型决策现已从 `src/spmvPlan.cu` 拆到 `src/selection/spmvPlanModel.cpp`，只返回决策和预测，不拥有 GPU 数据；`tools/plan_model_replay.cpp` 在 CPU 侧重放该决策。使用重构前这 40 张数据对原模型和 vector 模型各重放一次，80 次均与冻结的 GPU Plan 选型一致；当模型保持默认 AtBSR 时，只核验回退 backend，不核验由 AtBSR 自身模型选择的 shape。[回放审计](../results/model/selection-model/h100-vector-paired40/model-refactor-audit.json)。

新增可选 `model_refine` 对模型候选、默认 AtBSR 和六种 CSR vector 做有界实测，可选再加 mixed-tile，不能保证全局最优。其配对测试见 `results/model/selection-model/h100-refined-v3-paired40/`，GPU 被其他任务使用时会等待；在结果完成前不声称它有性能收益。

对同一轮 `auto_tuned_mixed` 初始化记录做事后候选覆盖分析：只从六个 CSR vector 中取最快，与全部初始化候选的最小中位数比较，40 张中 29 张在 5% 内；从六个 CSR vector 和五个 mixed-tile 共同取最快则为 34 张。两种构建顺序共 80 次中，初始化最快候选分别来自 CSR vector 56 次、单一 AtBSR 11 次、DSM/AtBSR 7 次、mixed-tile 6 次。此统计直接从同一轮初始化计时选最小值，不包含模型候选与默认 AtBSR 的补充覆盖，也没有独立复测、选型门槛或构建成本；只能说明值得测试有界候选集，不能证明新模式的最终速度。[统计](../results/model/selection-model/h100-vector-paired40/refinement-coverage.json)。


## 未校准 Ada 上的有界实测

在未启用 H100 预测模型的 RTX 5880 Ada 上，使用相同 40 张清单，对 `model_refine=true`、`enable_mixed_tile=true` 的方案与既有完整 `Auto+tune` 做两次反向构造顺序、预热 Graph 的配对测量；40 张均通过正确性检查。这里无预测候选，精调集合包含默认 AtBSR、六个 CSR vector 和五个 mixed-tile。完整调优不含 mixed-tile，因此两边候选集不同。

精调方案相对完整调优的加速比几何平均为 **1.015×**，36 / 40 张不慢超过 5%，1 张慢超过 10%；相对默认 AtBSR 几何平均为 1.073×。这不能证明精调比完整调优本身更准确：增加 mixed-tile、5% 切换门槛、各 Plan 的独立分配和 Graph 以及时间噪声都可能影响差异。每类 80 次构造记录的初始化中位数，精调约 16.2 ms，完整调优约 83.2 ms；它们是各自中位数而非逐矩阵成本差，且依赖当前机器和预热条件。样本较小、已被反复查看，不是盲测。[原始结果](../results/model/selection-model/ada-refined-paired40/analysis.json)。

在这 40 张已观察样本上，进一步用冻结模型预测回放 vector 切换门槛：门槛 0%、5%、10% 时，相对原模型的几何平均加速比分别为 1.117×、1.120×、1.085×；vector 选择次数分别为 30、26、17，慢于原模型超过 5% 的案例分别为 1、0、0。`bibd_17_4` 的 vector 预测优势约 3.7%，因此 5% 门槛在这个案例上回退原模型。该统计复用已测的两种 Plan 时间，门槛是事后尝试，并非独立验证；默认门槛没有据此修改。需要保守策略时可显式设置 `vector_model_min_time_reduction=0.05`，但不能保证新矩阵不退化。[回放统计](../results/model/selection-model/h100-vector-paired40/vector-margin-retrospective.json)。

## H100 有界精调：已查看的 40 张

H100 2 号卡空闲后，排队的精调对照已经完成，全部 40 张通过两组输入的直接校验和 Graph 校验。两种构建顺序下各比较模型、vector 模型、有界精调、含混合 tile 的完整调优、无混合 tile 的完整调优、默认 AtBSR 和 cuSPARSE；共七个已选 Plan。这是前述已查看过的 40 张，不能作为阈值或策略的盲测。

模型、vector 模型、有界精调相对七个已测 Plan 中最快者的时间几何平均分别为 1.164×、1.043×、1.016×；各自有 9、30、39 张处于 5% 以内。有界精调有 1 张慢超过 10%。这里的最优仍是七个选定 Plan 的事后最快，不是全部 kernel 候选的全局最优；图间地址、缓存和测量噪声仍可能影响结果。[结果](../results/model/selection-model/h100-refined-v3-paired40/oracle-analysis.json)。

基于相同配对记录估算 `Plan 初始化 + N 次预热 Graph SpMV`：有界精调相对纯 vector 模型在 N=10,000 时几何平均总耗时加速比为 0.609×，N=100,000 为 0.927×，N=1,000,000 为 1.014×。这只是该样本上的配对估算，不包括读取矩阵、调用方分配和 Graph 创建；首次上下文准备、地址及设备状态不同都会改变摊销点，不应当当作通用迭代次数阈值。[摊销统计](../results/model/selection-model/h100-refined-v3-paired40/refined-amortization.json)。

项目目录整理后的当前代码，另从原 389 张清单中随机取 40 张、排除上述 40 张，已在 `h100-refined-new40` 完成同口径复测。该排除仅按矩阵名称，不保证矩阵族或训练数据独立。[进度](../results/model/selection-model/h100-refined-new40/runner.log)。


## 当前代码：另一批 40 张 H100 矩阵

40 张均通过检查。七个已选 Plan 中取每种顺序实测最快者，模型、vector 模型、有界精调的运行时间比值几何平均分别为 **1.136×、1.061×、1.033×**；距该对照不超过 5% 的矩阵分别为 **8、25、32 张**；慢超过 10% 的分别为 **27、10、6 张**。相对 vector 模型，有界精调运行时间的加速比几何平均为 1.027×；相对有界精调，含混合 tile 的完整调优选出的方案几何平均快约 2.3%。[逐矩阵数据](../results/model/selection-model/h100-refined-new40/oracle-analysis.csv)。

6 张精调慢超过 10% 的矩阵，在两种顺序下的 12 个实测最快 Plan 中，6 次为 DSM/AtBSR 混合方案、6 次为单一 AtBSR。这与当前有界精调只测默认 AtBSR、模型选中的方案、六个 vector 和五个 mixed-tile 的候选范围一致：它可能漏掉其他 DSM 配置或单一 tile。两种独立 Plan/Graph 的计时也存在分配和噪声差异，因此这些记录说明了候选覆盖不足的可能方向，并非严格的单因素因果证明。

Plan 初始化时间各自的中位数：vector 模型约 4.11 ms，有界精调约 19.94 ms，含 mixed-tile 的完整调优约 209.19 ms；这些不是配对差值。按每矩阵、每种构造顺序估算 `准备 + N × 暖 Graph SpMV`，有界精调相对 vector 模型在 N=10,000、100,000、1,000,000 时的总时间加速比几何平均为 **0.664×、0.956×、1.019×**；N=1,000,000 时 21 / 40 张收益。估算不包含输入读取、调用方分配、Graph 创建和冷缓存，样本及候选选择也影响结果。[配对摊销结果](../results/model/selection-model/h100-refined-new40/refined-amortization.json)。

复现统计：

```bash
python3 scripts/analysis/analyze_plan_selection.py results/model/selection-model/h100-refined-new40
python3 scripts/analysis/analyze_selection_oracle.py results/model/selection-model/h100-refined-new40
python3 scripts/analysis/analyze_refined_amortization.py results/model/selection-model/h100-refined-new40
```

## 实验性 vector 预测模型与全量测试

为减少 CSR vector 宽度选择误差，实验模型在原有特征外加入六种 tile 的数量和最大数量，以及 128、256、4096 行长度阈值的尾部统计。模型预测各宽度相对默认 AtBSR 的时间，仍由统一 Plan 的既有门槛与其他候选比较。该模型仅在 `model_select`、`vector_model_select` 和 `vector_model_v2` 均为 true 时启用；默认行为不变，适用硬件和矩阵范围也不变。

基于已有 H100 调优记录，按近似矩阵族排除已查看的配对样本，再按族划分训练、开发和保留集：分别为 1130、455、245 张。按开发集选择的模型在保留集上的时间比预测平均相对误差为 7.60%；相同训练划分下使用原特征的对照模型为 8.93%。这是相对时间比的误差，而非完整 Plan 最优方案命中率，也不是与已部署模型的直接比较。[训练报告](../results/model/selection-model/vector-accuracy-v1/report.json)。

复用已测的独立候选时间回放，在 120 张样本上，实验模型与旧模型距离默认 AtBSR 和六个 vector 宽度中最快者不超过 5% 的分别为 115 和 104 张；对应几何平均耗时比分别为 1.006 和 1.017。这些样本已被查看，且近似族划分无法保证完全独立，不能代替新一轮实际 Plan 测试。

对回放中新旧模型的逐矩阵加速比按近似矩阵族做 5000 次簇 bootstrap：80 张样本的几何平均加速比为 1.0088×，95% 区间为 1.0001–1.0204×；120 张样本为 1.0107×，区间为 1.0044–1.0181×。[计算结果](../results/model/selection-model/vector-accuracy-v1/replay-uncertainty.json)。区间只反映该族划分下的重采样变化，不覆盖计时噪声、模型探索或其他训练数据重叠，不能当作全数据集或未见矩阵的保证。

C++ 导出的 10980 次预测与冻结的训练模型最大相对误差为 2.22e-16；100 张矩阵的运行时特征与训练输入逐项一致。Ada 上的 Plan 正确性测试通过 464 项直接、Graph 和参考值检查；H100 上新模型的设备正确性和性能仍需以后台任务为准。

静态数据清单共 2888 张：2248 张位于模型校准范围、561 张超出范围、49 张为当前读取接口不支持的复数矩阵、30 张缺少可靠结构统计。[完整清单](../results/model/selection-model/full-inventory/inventory.csv)。其中七张原始目录中的文件无效，但本机另一个数据目录的副本通过了结构检查；范围外测试使用软链接覆盖目录读取这些副本，不修改原文件。[副本检查记录](../results/model/selection-model/unknown-scope-audit/alternate-structure.csv)。现有 H100 基线测试覆盖 2247 张；实验模型的 2248 张对照在另一张同规格 H100 上并行运行。范围外矩阵及缺少统计的矩阵尚未完成性能评估，不能把这次扫描称为 2888 张全量性能结果。

该 2247 张基线运行已完成，对比同轮 cuSPARSE 的预热 Graph 时间，原统一模型、CSR vector 模型、有界实测选型和混合方案初始化调优的几何平均加速比分别为 1.426×、1.565×、1.596× 和 1.625×，分别有 2103、2147、2196 和 2227 张快于 cuSPARSE。统计由 [逐矩阵结果](../results/model/selection-model/h100-full-2247/analysis.csv) 的配对比值计算；加速比是 cuSPARSE 耗时除以方案耗时，两种构造顺序的比值先按几何平均合并，再对矩阵取几何平均。计时包含 Graph 内事件节点开销，不包含 Plan 初始化、矩阵读取及 Graph 创建；初始化调优的准备代价未计入。不同 Plan 的独立分配和构造顺序可能影响结果，尚不能代表范围外矩阵的表现。

561 张已知范围外矩阵的对照任务也已排队：模型不在这些矩阵上预测，因此测量的是有界实测选型与完整调优、含 mixed tile 的完整调优、默认 AtBSR 和 cuSPARSE 的表现。调优可能跳过无法分配的候选或遇到资源限制；原始日志和状态必须一并检查。两种构造顺序并不能消除所有地址和缓存差异。[范围外任务](../results/model/selection-model/h100-outside-scope-550/run.sh)。

若范围外完整对照明确因 GPU 显存不足、tile 偏移上限或超时失败，将在前序任务结束后尝试一个较小的候选集：默认 AtBSR 与六种 CSR vector 宽度，并单独比较 cuSPARSE；原始失败状态保留。它不包含 mixed tile、DSM 或完整调优，结果记录在独立的 `resource_retry_*` 字段中，不能与完整候选集合的最优值直接比较。数值错误及原因不明的失败不会自动归入资源重试。[重试任务](../results/model/selection-model/h100-resource-retry/run.sh)。

重试清单冻结原始状态与日志的 SHA-256；汇总时还会比较两次测试的矩阵哈希。原始证据或矩阵内容发生变化的重试会分别标为 `source_evidence_changed` 或 `matrix_changed`，不计入性能均值。

三项任务的逐矩阵状态会定期汇总到 [全数据集记录](../results/model/selection-model/all-datasets-summary/all-datasets.csv) 与 [统计](../results/model/selection-model/all-datasets-summary/summary.json)。每个任务比较的候选集合不同，其实测最快者不能直接混为统一的全局最优；失败、待测和不适用的矩阵不会计入性能均值。

逐矩阵汇总同时保留相对默认 AtBSR、cuSPARSE 的加速比；实验模型运行开始后，`v2_paired_speedup_vs_v1` 采用同一次运行、相同构造顺序中的 v1/v2 配对计时。不同批次独立测得的几何均值不应直接当作配对比较。

逐矩阵汇总也会合并绝对参考模型回放中的训练划分、预测和实测参考时间及误差；尚未完成 v2 测试或尚未完成回放的条目分别标为 `pending` 和 `not_evaluated`。该回放使用冻结模型在日志上离线评估，不代表这轮测试直接执行了新接入的 Plan 选项。

汇总中的 `invalid_input` 与 `unsupported_int32_nnz` 会根据扫描进展从静态清单的 `unknown_structure` 中单独列出；`input_source` 记录是否使用了本机恢复的副本。静态清单保留最初的范围划分，动态分类以逐矩阵汇总为准。

静态清单中仍缺少精确统计的 30 张矩阵，按扫描和文件大小检查分为：1 张扫描发现数据无效，2 张的文件长度不足以容纳声明的坐标数，2 张声明的非零数超出当前 32 位 CSR 表示范围，另 25 张正在逐文件扫描。文件大小检查未读取完整的大文件，不能用来证明文件有效。[输入状态](../results/model/selection-model/unknown-scope-audit/input-status.csv)。

剩余 25 张的低优先级逐文件结构扫描已完成：21 张结构有效且可用 32 位索引，4 张展开后超出 32 位非零数上限。[扫描日志](../results/model/selection-model/unindexed-25-scan/runner.log)。前一批 H100 测试成功结束后，21 张将进入 [追加测试](../results/model/selection-model/h100-unindexed-extra/run.sh)。汇总表根据扫描结果动态更新类别，因此其数字不同于上面的静态清单；结构有效不保证矩阵及所有候选能放进设备显存。

新旧 vector 模型在同一次 H100 运行中配对测量，并会按实验模型的训练、开发、保留和未列入训练表分组，写入 `h100-v2-full-2247/v2-paired-analysis.json`。这些分组仅描述 vector 模型的数据划分；统一模型的其他组成部分可能存在训练数据重叠，既有结果也被查看过，不应当作完全独立的盲测。

该配对分析还分别记录两种 Plan 的初始化时间，并估算不同调用次数下 `初始化 + N 次预热 Graph` 的总耗时比。初始化差异同时包含候选构造、内存分配及选择方案不同带来的影响，不能等同于预测函数本身的开销；估算也不包括读取矩阵、调用方分配或 Graph 创建。

这轮 v2 实测还会记录各模型候选的预测时间比和绝对时间估计，并对最终选中且能匹配到预测记录的配置计算误差。相对时间以独立计时的默认 AtBSR 为分母，绝对时间使用模型内部的参考时间；回退到默认方案且没有对应预测的样本不计入预测误差。Graph 内事件开销、独立 Plan 分配和不同时间来源都会影响对照，误差不能直接等同于纯模型误差。

预测误差会按非零数区间、最终后端和参考时间是否经过校准分组，同时报告每组参与误差计算的样本数。分组有选择偏差：只比较实际选中的、可以匹配预测记录的候选，不能代表所有候选配置的预测精度。

早期 v2 实测发现，H100 的默认 AtBSR 绝对参考时间未校准，直接用于 `estimated_us` 的误差较大。为探索改善，使用 2 号 H100 已完成的 1010 张训练矩阵拟合一个实验性绝对参考模型；225 张同卡保留矩阵上的平均相对误差为 6.18%。在另一张 H100 的一次 153 张中间统计中，20 张保留样本的原参考误差为 42.5%，实验参考误差为 6.23%；将相对预测乘以该参考后，六种 vector 宽度的绝对时间平均相对误差从 39.3% 降到 5.11%。[训练报告](../results/model/selection-model/h100-absolute-reference-v1/report.json)；[持续更新的跨卡评估](../results/model/selection-model/h100-absolute-reference-v1/reference-evaluation.json)。

这里的保留集按近似矩阵族划分，两个设备共用主机，结果和模型选择已经被查看；计时包含 Graph 内事件节点，可能不同于实际应用的计时方式。两张 H100 的并行任务也可能共享主机或 PCIe 资源，即使各自监控 GPU 占用，仍不能完全排除这种干扰。

持续更新的跨卡报告还包含中位数、90 分位、最大误差和比原参考更差的案例。在一次 459 张的中间统计中，实验参考有 5 张平均相对误差超过 20%，其中 `G2_circuit` 的误差约 41.8%，而原参考约为 5.7%。模型并不保证逐矩阵改善，当前仍应将它作为显式实验选项。

针对尾部误差做了一次回放探索：用森林各树的对数时间预测标准差作为不确定性启发，在开发集上选择回退到原参考的阈值。在已查看的 53 张保留样本中，这一阈值使 1 张回退，最大误差从 52.4% 降到 18.1%，平均误差从 7.39% 降到 6.66%。[冻结回放](../results/model/selection-model/absolute-uncertainty-exploration/report.json)。树间分歧不是经过校准的置信区间，另一些高误差案例的树间分歧很小；样本小且已被观察，回退策略尚未接入 Plan。

将该已冻结阈值应用于随后完成、未出现在上述回放快照中的 86 张矩阵，平均误差从 6.11% 增至 6.73%，超过 20% 的案例从 4 张增至 6 张，最大误差仍为 62.0%。[后续样本回放](../results/model/selection-model/absolute-uncertainty-followup/report.json)。这批样本仍可能与历史数据共享矩阵族，但结果已足以说明该门槛缺乏稳定收益，不应直接启用。

另对四张大误差矩阵及六张训练样本提取列索引结构，检查输入向量访问局部性是否值得进一步研究。[结构统计和输入哈希](../results/model/selection-model/reference-locality-sample/locality.json)。例如，`delaunay_n17` 的平均行内列跨度约 55388，而训练样本 `cont-300` 约 46046；两者相近，因此不能仅凭跨度解释前者的预测偏差。统计基于 SciPy 合并重复项后的 CSR 和假定对齐的 FP64 向量，并不反映 AtBSR 的实际访问顺序或缓存命中；尚无证据证明缓存导致了误差，也未据此修改选择模型。空行和重复项的边界处理通过 [合成样例](../tests/x_locality_sample_test.py) 核验。

检查高误差矩阵后，增加了事后定义的结构分组：非零数不少于 600,000 且最长行小于 32。当前冻结的 1010 张参考模型训练集中只有 6 张满足该条件；一次中间统计的 7 张此类实测矩阵中，实验参考平均误差为 35.8%，原参考为 16.7%。该范围和阈值是查看结果后确定的，样本很小，不能据此断言误差原因或直接设置选择规则；它提示后续应增加此类训练与独立验证样本。分组将随 [跨卡评估](../results/model/selection-model/h100-absolute-reference-v1/reference-evaluation.json) 更新。

实验参考已接入为默认关闭的 `SpmvPlanOptions::absolute_reference_model`，须同时启用 `model_select`；模型仅在现有 H100 PCIe 校准范围内运行，可通过 `info().absolute_reference_model_used` 检查。它替换 `model_reference_us` 和各候选的 `estimated_us`，候选的相对时间比及选择门槛不变。为避免把实验拟合误认作硬件解析模型校准，使用该参考时 `model_reference_calibrated` 为 false。范围外保留原有回退行为。该选项针对本实验的暖 Graph 计时，尚未完成全量跨卡验证，不能据此保证生产环境的绝对时间准确性。

冻结模型导出后，1646 条特征输入的 C++ 与训练模型最大相对误差为 8.88e-16；当前跨卡记录中的 41 张保留矩阵和 56 张未列入训练表的矩阵也通过了读取真实矩阵的主机端比较，选择配置未改变，预测参考值一致。[保留集核验](../results/model/selection-model/h100-absolute-reference-v1/host-plan-audit.json)；[未列入训练表的样本](../results/model/selection-model/h100-absolute-reference-v1/host-plan-audit-excluded.json)。Ada 上的 Plan 正确性检查通过；H100 上选项本身的设备检查仍在队列中。

补入的 Oregon-1 已单独提取四个阈值的特征，并与 C++ 运行时特征逐项核对；评估时使用冻结的矩阵哈希核验。当前两张 H100 若有其他计算进程，运行器会等待空闲并丢弃检测到冲突的未完成案例；这仍不能排除轮询间隔内的短暂干扰。

v2 全量测试另外独立构造并计时六种 CSR vector 宽度，从而计算所有宽度的相对和绝对时间预测误差，以及预测最快宽度相对六个实测宽度的差距。它们是独立分配的 Plan，可能受地址和计时噪声影响。加入这六个候选后，v2 轮次的“已测最快”候选集合比基线轮次更大；应优先使用同一次 v2 轮次内的新旧模型配对结果，不能直接比较两轮的最优命中率。

对运行中一次 758 张的快照按名称近似矩阵族进行 5000 次配对 bootstrap：全部样本的新旧 Plan 几何平均加速比为 1.0036×，重采样区间为 1.0015–1.0059×；93 张 vector 模型保留样本为 1.0015×，区间为 0.9964–1.0081×。六种宽度的平均相对时间预测误差在全部样本中从 8.21% 降至 6.04%，但保留样本两者均约 7.78%，目前不能据此断言保留集上有稳定改进。[快照与计算结果](../results/model/selection-model/v2-bootstrap-prefix/report.json)。这些区间仅描述已完成前缀中基于名称分族的重采样变化，不包含计时噪声、模型探索、多次查看结果或数据集抽样偏差，不能推广为全量结果的置信保证。

后续 2248 张校准范围内的 v2 对照已完成（运行退出码 0），六种 vector 宽度的平均相对时间预测误差从旧版 9.12% 降至新版 6.45%；其中 245 张近似按矩阵族划分的保留样本从 8.15% 降至 7.86%，418 张未列入训练表的样本从 11.58% 降至 9.42%。实际选中 Plan 的同轮新旧几何平均加速比分别为全体 1.0070×、保留样本 1.0037×。全体 2248 张中，新版所选 Plan 的预热 Graph 耗时相对本轮已测最快候选的几何平均比值为 1.0427，1740 张差距在 5% 以内；相对同轮 cuSPARSE 的几何平均加速比为 1.5939×。[完整配对结果](../results/model/selection-model/h100-v2-full-2247/v2-paired-analysis.json)。这些分组仅针对 vector 模型，其他模型组件或先前实验可能使用过相关数据；实际加速不包含初始化。

以这次独立构造的六种宽度 Graph 测量作为新训练标签做离线探索，复用原有特征及矩阵族划分（训练 1130、开发 455、保留 245），并核对全部输入矩阵哈希与历史调优记录一致。四组树参数中，深度 14、叶子最少 3 个样本的候选在开发集的宽度时间比平均绝对相对误差为 6.61%，冻结 v2 在相同口径下为 6.98%；保留集分别为 7.57% 和 7.83%。[实验结果](../results/model/selection-model/vector-graph-label-exploration/report.json)。两种口径均先对两种构造顺序的时间比取几何平均，再计算误差，与上面逐顺序求误差的统计不完全相同。该实验已查看保留集，候选间差异较小且共享主机和历史数据，尚不足以证明泛化改善。

同一候选在另外 418 张未列入旧训练表的矩阵上，平均误差为 8.95%，冻结 v2 为 9.39%；在六个已测宽度中，预测宽度距最快者不超过 5% 的矩阵分别为 405 和 395 张。此处的标签仍来自同一次 H100 测试，且这些矩阵在先前实验中可能被查看过，不能作为完全独立的最终验证。额外矩阵的特征和输入哈希与冻结特征来源逐项核对；Oregon-1 使用单独保存的输入哈希。

对同一候选与冻结 v2 的逐矩阵平均误差差值按近似矩阵族做 5000 次配对重采样：保留集误差降低 0.255 个百分点，区间为 -0.037 至 0.591 个百分点，245 张中有 127 张误差反而增大；未列入旧训练表样本降低 0.441 个百分点，区间为 0.293 至 0.592 个百分点，418 张中有 170 张误差增大。区间仅反映按名称分族的样本重采样，不覆盖计时噪声、模型参数探索或数据集选择偏差，不能代替独立实验；保留集结果尤其不能证明可靠改善。

为测试实际 Plan，已将这个候选导出为默认关闭的 `vector_model_graph` 选项；它与 `vector_model_v2` 互斥且需要 `vector_model_select`。10980 条输入上 C++ 导出与 sklearn 最大相对误差为 2.22e-16，Ada 上 Plan 正确性检查通过。H100 的实际 Plan 正确性及 663 张保留/旧训练表外矩阵的同轮对照已排队，测试会等待设备空闲；[运行脚本](../results/model/selection-model/h100-vector-graph-663/run.sh) 与模型及可执行文件的哈希已冻结。该组矩阵和历史数据已被观察，不能称为全新盲测；在对照完成前，不应推断实际性能改善。

H100 上的新选项已通过 952 项直接执行、Graph 和参考值检查（测试中检测到其他 GPU 活动后重新运行）；663 张同轮性能对照结果由独立收尾脚本生成。

663 张保留及旧训练表外矩阵的同轮对照已完成，全部状态为成功，输入哈希与原 v2 测试一致。六种宽度相对时间比的平均误差由 v2 的 8.75% 降至 Graph 模型的 8.65%；Graph 模型实际选中 Plan 相对 v2 的几何平均加速比为 1.0006×，13 张快超过 5%，9 张慢超过 5%。[同轮结果](../results/model/selection-model/h100-vector-graph-663/graph-comparison.json)。预热 Graph 不含初始化，模型间各自分配的宽度候选和事件节点可能影响计时；这些数据已被查看过。当前证据表明平均误差和性能差别都很小，不足以支持将 Graph 模型改为默认。

为覆盖全部 2248 张模型范围内矩阵，另将 1585 张训练/开发样本排在上述 663 张之后执行，使用相同冻结二进制和计时配置；两组分别报告，全部完成后再合并为 `h100-vector-graph-all/comparison.csv` 与 `summary.json`。训练/开发结果包含模型见过的数据，不能用于证明泛化；663 张也被先前实验查看过，仍不属于完全盲测。实际测试若有失败或跳过，合并结果会保留各组状态计数，不把缺失值当作成功测量。

2248 张对照现已全部成功，输入哈希与旧 v2 运行一致。全部样本的六宽度时间比平均误差从 6.38% 降到 6.29%，新方案相对 v2 的几何平均加速比为 1.0007×，45 张快超过 5%，27 张慢超过 5%；相对 cuSPARSE 为 1.5678×。这包含 1585 张训练/开发数据，且只比较预热 Graph 执行，不包括初始化和预测成本，因此不表明新模型的泛化能力或端到端收益。[汇总](../results/model/selection-model/h100-vector-graph-all/summary.json)、[逐矩阵对照](../results/model/selection-model/h100-vector-graph-all/comparison.csv)、[输入核验](../results/model/selection-model/h100-vector-graph-train-dev/input-audit.json)。保留/旧训练表外的 663 张单独结果仍更适合评估选择变化，但该组也已被查看过；目前不建议默认启用新模型。

[全数据集逐矩阵总表](../results/model/selection-model/h100-vector-graph-all/all-datasets-combined.csv) 将 2888 张的范围、原有选择结果和 Graph 对照合并；`pending_analysis` 表示测试已成功但该批次尚未完成汇总，`not_measured` 表示该矩阵没有纳入 Graph 对照，并不代表测试失败或性能相同。后台 `atspmv_joined_summary` 周期刷新，并在相关测试和最终汇总结束后再次刷新；结果仍需检查状态和输入审计。

对新旧两轮实际执行的输入增加了 SHA-256 核验：[已完成前缀检查](../results/model/selection-model/h100-vector-graph-663/input-audit.json)；全部测试结束后由独立后台任务检查两组全部矩阵并写入 `h100-vector-graph-train-dev/input-audit.json`。该核验仅确认输入内容相同，不能证明数值结果或排除测量干扰。

同一批 2248 张的冻结绝对参考模型离线评估中，旧/新参考平均绝对相对误差为 37.85% / 5.24%；245 张保留样本为 40.44% / 6.57%。新版仍有 73 张误差超过 20%，最大误差为 61.98%。[完整参考评估](../results/model/selection-model/h100-absolute-reference-v1/reference-evaluation.json)。可选绝对参考模型的 H100 Plan 正确性测试也已通过 948 项直接执行、Graph 和参考值检查，但上述误差仍是对另一次测量记录的离线评估，不等同于实际调用的端到端误差。

对基线运行的有界精调失误做逐配置检查：在一次 556 张的中间统计里，37 张相对七个已测 Plan 的最快者慢超过 10%；这 37 张的两种构造顺序共 74 个最快配置都未出现在该次有界精调的候选日志中，其中 48 个为 DSM/AtBSR 混合、26 个为单一 AtBSR。这支持进一步扩大候选覆盖的方向；计时噪声和各 Plan 的独立分配仍可能影响结论，不能单凭候选缺失断言因果。此统计是随机前缀，不代表全量结果。[随运行刷新的分析](../results/model/selection-model/h100-full-2247/refined-misses.json)。

为研究行长度在 128–1023 的 DSM 候选，使用历史全调优日志训练了探索性时间比模型：输入为 128/1024 阈值的结构统计、tile 和 cluster 大小；目标是相对该次调优中最快的非混合候选的时间比。按此前近似矩阵族划分，具有中等长度行的训练、开发、保留样本分别为 254、72、38 张。38 张保留样本的 12 个候选时间比平均相对误差为 6.65%；按预测选择候选并应用 5% 门槛后，32 张距回放最快候选不超过 5%，2 张慢超过 10%。[探索记录](../results/model/selection-model/medium-row-model-exploration/report.json)。

这里的基准是事后已知的最快非混合候选，而实际选择器无法免费获得这个基准；标签来自初始化调优计时，也不是新 Plan 的独立执行时间。样本数量小且已用于探索，因此这些结果只能说明后续建模方向，不能证明统一选择性能提升；该模型尚未接入 Plan。

进一步将中等行候选的目标改为相对默认 AtBSR 的时间比，并按现有统一模型和 vector-v2 的预测结果及混合方案 10% 门槛进行回放。38 张保留样本中，预测选择了 10 张中等行混合候选；回放相对现有选择的几何平均加速比为 1.058×，10 张快超过 5%，没有慢超过 5% 的样本。现有选择的 Python 回放与 C++ 主机选择器在这 38 张上的配置和预测值一致，矩阵文件哈希也与历史调优记录一致。[回放](../results/model/selection-model/medium-unified-exploration/report.json)；[C++ 对照](../results/model/selection-model/medium-unified-exploration/cpp-parity.json)。

回放仍混用了初始化计时和独立 Graph 计时，且样本已被查看；这些数字不是实测的新模型加速比。为验证实际行为，已排队在 H100 上逐张独立构造并测量现有 vector-v2 Plan 与模型挑出的一个 DSM/AtBSR Plan，包括两种构造顺序及数值校验。[测试任务](../results/model/selection-model/h100-medium-holdout38/run.sh)。

该实验还会检查输入哈希和基线实际配置是否与冻结回放一致，并记录 `Plan 初始化 + N 次预热 Graph` 的估算；估算不包含中等行模型的预测成本、输入读取或 Graph 创建，且不同 Plan 的构造顺序和分配可能影响结果。测试结果写入 `h100-medium-holdout38/analysis.json`，仅比较现有方案与每张矩阵的一个预测候选，并非所有配置的全局最优。

H100 实测现已完成：38 张均成功，冻结回放选择中等行混合候选的 10 张相对现有 vector-v2 Plan 均快超过 5%；38 张整体的预热 Graph 几何平均加速比为 1.0528×，没有慢超过 5% 的案例；29 张距离这两个实际测量 Plan 的较快者不超过 5%。[逐矩阵结果](../results/model/selection-model/h100-medium-holdout38/analysis.csv) 和 [汇总](../results/model/selection-model/h100-medium-holdout38/analysis.json)。这是此前已查看过的 38 张样本，选择结果来自冻结回放，不能视为对新矩阵的盲测；对照只包含现有 Plan 和一个预测的混合 Plan，不是所有配置的全局最优。运行时间未计初始化及预测成本，探索性中等行模型仍未接入生产 Plan。

查看进度与中间结果：

```bash
tail -n 20 results/model/selection-model/h100-full-2247/runner.log
python3 scripts/analysis/analyze_selection_oracle.py results/model/selection-model/h100-full-2247
tail -n 20 results/model/selection-model/h100-v2-full-2247/runner.log
tmux list-sessions
```

上述 oracle 是每种构造顺序中若干已测 Plan 的最快者，不是所有 kernel 配置的理论最优。Graph 预热时间不包括 Plan 准备成本；任务未结束时的统计只描述已完成的随机前缀。

后台收尾任务 `atspmv_final_summary` 等待周期汇总与所有测试任务结束后再执行两次汇总：第一遍更新绝对时间评估，第二遍将其合入总表，随后冻结 v2 配对数据并生成按矩阵族重采样的报告。收尾日志为 `results/model/selection-model/all-datasets-summary/finalize.log`；收尾完成只表示汇总步骤运行完毕，仍须检查各运行的退出状态、审计结果和数据集的 pending/failure 分类。

按一次 1225 张的中间统计估算 `Plan 初始化 + N 次预热 Graph`，有界精调相对纯 vector 模型在 N=10,000、100,000、1,000,000 时的几何平均总耗时加速比分别为 0.661×、0.946×、1.009×。这只是逐矩阵和构造顺序的配对估算，不包括输入读取、调用方分配和 Graph 创建，也不是可推广的调用次数阈值。[持续更新的估算](../results/model/selection-model/h100-full-2247/refined-amortization.json)。
