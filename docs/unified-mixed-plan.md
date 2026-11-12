# 在 SpMV Plan 中使用混合 tile

## 固定稀疏结构下更新数值

迭代仿真中，矩阵可能保持相同的 CSR 偏移、列索引和条目顺序，而数值随时间变化。
构建时显式开启 `enable_value_updates=true`，并检查 `info().value_updates_enabled`，
即可使用 `updateSpmvValues(plan, device_values, stream)` 更新数值。`device_values`
按最初输入的 CSR 条目顺序排列，必须在异步更新结束前保持有效。调用者负责保证
结构未改变；若偏移、列索引、维数或条目顺序改变，必须重新构造 Plan。

AtBSR 复用原行排序和 tile 偏移，直接将新 CSR 数值打包到既有槽位；mixed-tile
复用原行分组和 shape；CSR/cuSPARSE/CSR DSM 路径更新已有的值数组。混合 DSM/AtBSR
暂不支持，`value_updates_enabled` 返回 false。默认不开启，不增加旧用法的常驻 CSR 偏移存储。
该接口不重新选型、不分配、不做主机同步，支持 Graph capture；更新和执行应在同一
stream 串行使用，或由调用者显式建立依赖。更新会影响已有捕获图后续执行看到的矩阵。

可选的 MFEM 热传导应用会在更新前比较完整 CSR 偏移及列索引，结构一致时更新数值，
否则重建 Plan，见 [MFEM 接入说明](../apps/mfem/README.md)。这减少的是结构分析和转换成本，
并不保证矩阵乘或整个应用更快；应用收益须包含更新、组装及预处理器的实际耗时。

混合 tile 原型已提取到 `include/detail/mixedTile.cuh`，实验程序和 Plan 使用同一套打包与 kernel 实现。其选型目前仅支持初始化实测；H100 的 AtBSR/DSM 预测模型没有新增未经验证的混合候选，默认 Plan 行为保持不变。

```cpp
atspmv::SpmvPlanOptions options;
options.backend = atspmv::SpmvBackend::Auto;
options.tune = true;
options.enable_mixed_tile = true;
options.fast_tune = true;  // 可选：使用现有有界候选集
options.hybrid_min_time_reduction = 0.05;
auto plan = atspmv::makeSpmvPlan(device_csr, options, stream);
atspmv::executeSpmv(plan, x, y, stream);
```

Auto 调优会在原有 AtBSR、CSR/DSM、混合 DSM/AtBSR 等可用候选之外，加入混合 tile 的五种执行方式：动态 shape、特化 shape、以及每任务最多 4、8、16 个 tile 的 balanced 版本。五者共享同一套混合打包数据。实测使用现有 Plan 的预热 CUDA Graph 方法，包含执行阶段的清零、原子加及多个 kernel，不包含构建、实测和 CPU 调用开销。`hybrid_min_time_reduction` 对混合 tile 和已有 DSM 混合候选生效：只有达到相对于最佳其他候选的时间下降幅度才会切换；显式选择某个 backend 时不设置这个门槛。开启选项但 `tune=false` 且 `model_refine=false` 不会加入混合 tile，也不会修改模型选择行为。

也可以显式使用混合 tile：

```cpp
atspmv::SpmvPlanOptions options;
options.backend = atspmv::SpmvBackend::MixedTile;
options.mixed_tile_cap = 8; // -1 动态；0 特化；4/8/16 是 balanced 的任务上限
auto plan = atspmv::makeSpmvPlan(device_csr, options, stream);
```

显式设置 `backend=MixedTile` 和 `tune=true` 时，Plan 实测五种执行方式并选择最快者。`plan.info().backend`、`mixed_tile_cap`、`timings`、`setup_us` 和 `skipped_candidates` 分别记录赢家、配置、每个候选的时间、初始化总耗时以及被跳过的候选；每项混合候选的 `mixed_tile_cap` 也会记录在 `timings` 中。Auto 构造混合布局遇到 32 位槽偏移溢出或 CUDA 显存不足时，跳过混合候选并继续已有候选；显式请求混合时则返回错误。

可以通过 `spmv_plan_bench MATRIX.mtx --fast-tune-mixed` 做单矩阵冒烟检查，也支持 `--tune-mixed` 和 `--mixed`。Plan 构建会把 CSR 数据复制到主机构建布局，并一次性建立几种任务描述；在大矩阵上可能很慢、占用额外内存。Plan 不知道后续调用次数，不会自动将构建与实测成本摊销，因此是否启用该选项应由调用者根据预期复用次数决定。

此版本尚未给混合 tile 建立 H100 预测模型、实现候选特征预筛选或实现 DSM 长行与混合 tile 短行的组合 backend。Auto 可并列比较现有 DSM 候选与完整矩阵的混合 tile 候选。H100 正确性检查及两个单矩阵冒烟选型已完成，结果见 `results/model/mixed-tile/unified-plan-h100-check/`；单次冒烟结果不能作为性能研究。


## 可选的模型引导实测

```cpp
atspmv::SpmvPlanOptions options;
options.model_select = true;
options.vector_model_select = true; // 预测候选包含 CSR vector
options.model_refine = true;       // 实测默认 AtBSR、预测候选和六种 CSR vector
options.enable_mixed_tile = true;   // 可选：增加五种 mixed-tile 执行方式
auto plan = atspmv::makeSpmvPlan(device_csr, options, stream);
```

`model_refine` 要求 Auto、`model_select=true`、`tune=false`，且不能强制 tile/mode。模型适用时，Plan 用已有的 Graph 测量流程比较默认 AtBSR、模型选出的方案和六种 CSR vector；重复候选会去重。`enable_mixed_tile` 再加入五种混合候选。它不自动枚举其他 DSM 或 AtBSR 配置，因而是有界候选集，不保证全局最优；对 DSM/混合 tile 沿用实测收益门槛。`info().model_refined`、`measured`、`timings` 和 `setup_us` 可检查实际行为及成本。模型范围外，没有预测候选时也会以默认 AtBSR 为基线实测 CSR vector 和可选 mixed-tile；若显式开启 `model_fallback_tune`，则改用已有的全量或 fast 调优回退。默认仍关闭精调，调用者应根据复用次数评估初始化成本。

命令行可使用 `spmv_plan_bench MATRIX.mtx --model-refine`，或者用 `--model-refine-mixed` 同时加入混合 tile。H100 和 Ada 配对结果与初始化摊销限制见 [选型验证](selection-model-validation.md)；排除前一批矩阵的当前代码复测也已完成，结果和限制见该报告。

## 可选的无损列索引压缩

```cpp
atspmv::SpmvPlanOptions options;
options.backend = atspmv::SpmvBackend::AtBsr;
options.tile_index = 5; // 32x1
options.balanced = 0;
options.compress_tile_columns = true;
options.enable_value_updates = true;
auto plan = atspmv::makeSpmvPlan(device_csr, options, stream);
```

`compress_tile_columns` 默认关闭，目前支持普通 `32x1`、原始 int32 列索引的 AtBSR。
GPU 对每个 32 槽 tile 计算列号范围：范围小于 255 使用 uint8 偏移，小于 65535
使用 uint16 偏移，其他 tile 保留 int32；额外的最小列号和位置元数据用于无损解码。
浮点数值、原始列号和 CSR 数值更新顺序保持不变。布局不支持、偏移可能溢出或压缩后
不节省空间时保留原表示；查看 `info().tile_columns_compressed` 和 `tile_column_bytes`。
压缩路径使用两路独立累加，不能要求逐位等同于串行加法。

压缩构建和清理均包含在 Plan 初始化成本内。现有预测模型没有为压缩布局标定；
此选项不会自动启用 X 的 L2 保留策略，后者由应用按 stream 和输入向量地址管理。
应用集成与独立性能结果见 [MFEM 应用](mfem-application.md)。

## 按索引宽度分组的原生 AtBSR 布局

```cpp
atspmv::SpmvPlanOptions options;
options.backend = atspmv::SpmvBackend::AtBsr;
options.tile_index = 5;
options.balanced = 0;
options.group_tile_columns = true;
options.enable_value_updates = true;
auto plan = atspmv::makeSpmvPlan(device_csr, options, stream);
```

`group_tile_columns` 与 `compress_tile_columns` 互斥，默认均关闭。前者在每个 M 行组内
将 tile 分为 8/16/32 位三个连续区间，分别保存定长列索引流和对应的 FP64 值。
列号位置由 tile 序号直接计算，基准和偏移可以并行读取，不需要先解码变长存储位置。
每个 warp 累加三个区间，再按 K=32/M 做行内归约，由每行首 lane 写回结果。
无新增原子加或 kernel 拆分，不配置 L2 策略。

布局支持六种普通 shape（`tile_index=0..5`），要求原始 int32 列索引；balanced、已有 int16 索引或不节省空间时
保留原实现。`info().tile_columns_grouped` 表示实际生效，`tile_column_bytes` 为运行时
列索引与区间/基准元数据大小。开启数值更新后，另保存 `tile_value_map_bytes` 大小的
来源映射，用一个 kernel 从原 CSR 顺序直接更新分组后的值数组，无需每步重新分组。

初始化全部在 GPU 完成，原来的列索引/数值暂存可在选择后释放。共享候选存储时保留
其原所有者，避免重复释放。分组改变浮点累加顺序，不保证逐位相同；必须按容差验证。
现有模型没有为此布局标定，不设置开关时，自动选中某个 shape 不会启用分组布局。
六种 shape 的性能与适用范围见 [扩展测试](grouped-columns-six-shapes.md)；原有 MFEM
32×1 应用结果见 [原生 AtBSR 优化](mfem-native-atbsr.md)。

## H100 冒烟验证

已停止原 GPU 3 大矩阵调优任务并在空闲 H100 上执行 `spmv_plan_test --quick`：DSM 可用，892 项直接调用、Graph 与参考值检查通过。两个单次 `--fast-tune-mixed` 冒烟案例均通过数值检查：`trans4` 选择 balanced mixed cap 16（7.604 μs，最快其他候选 cuSPARSE 8.184 μs），`rajat09` 选择特化 nonbalanced mixed（3.364 μs，最快其他候选 CSR vector 4.028 μs）。这些值是初始化调优中的单次结果、fast shortlist、warm Graph，并非独立重复性能评估；调优和构建耗时分别约 3.37 s 和 40.9 ms。完整日志见 `results/model/mixed-tile/unified-plan-h100-check/`。

新的 H100 80 张矩阵配对试验等待先前排队的 `fast-tune-paired80` H100 工作完成后启动，结果路径 `results/model/mixed-tile/h100-fair-cap-paired80/`。该实验比较混合与单一形状候选，不包含 DSM，不代表生产 Plan 自动选择性能。
