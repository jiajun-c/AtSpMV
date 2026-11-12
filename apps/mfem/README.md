# MFEM 热传导应用

此目录将 AtSpMV 接入 MFEM v4.8 的 `examples/ex16.cpp`，上游提交
`a01719101027383954b69af1777dc828bf795d62`：
<https://github.com/mfem/mfem/tree/a01719101027383954b69af1777dc828bf795d62>。
沿用 BSD-3-Clause 许可，原许可和通知保存在本目录。

应用求解 `du/dt = div((kappa + alpha*u) grad(u))`，采用有限元空间离散，
自然绝热边界，以及 MFEM 的隐式时间积分和 Jacobi 预处理 CG。质量矩阵为 M，
扩散矩阵为 K；隐式阶段求解 `(M + dt*K) k = -K*u`。

`spmvOperator.hpp` 将 MFEM 的矩阵乘接入现有 `SpmvPlan`，不修改 MFEM 库。
所有内核在同一 CUDA 默认 stream 上执行，目前仅支持串行 FP64 CUDA MFEM。
可选后端为 MFEM 原生、Plan cuSPARSE、原 AtBSR、自适应混合 tile，以及初始化实测选型。
`atbsr-compressed` 强制普通 `32x1`，在可节省空间时使用无损 8/16/32 位 tile 列索引。
`atbsr-grouped` 使用按索引宽度分组的定长流内核和直接数值更新，同样保留 FP64；
不自动开启 L2 保留，原理和对照见 [原生 AtBSR 优化](../../docs/mfem-native-atbsr.md)。
`atbsr-tuned` 在六种 shape × 两种模式内实测选择，`tuned` 使用 Plan 的有界 Auto 候选集。
不支持将已经组装的 CSR 后端直接用于 matrix-free/partial assembly。

## 构建

先编译 MFEM v4.8 的 CUDA、双精度、串行版本，关闭 MPI 和 METIS：

```bash
git clone --branch v4.8 --depth 1 https://github.com/mfem/mfem.git /tmp/mfem
cmake -S /tmp/mfem -B /tmp/mfem-build \
  -DCMAKE_BUILD_TYPE=Release -DMFEM_USE_CUDA=YES \
  -DCUDA_ARCH=sm_90 -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DMFEM_USE_MPI=NO -DMFEM_USE_METIS=NO \
  -DMFEM_ENABLE_EXAMPLES=NO -DMFEM_ENABLE_MINIAPPS=NO
cmake --build /tmp/mfem-build --parallel 8
cmake -S . -B /tmp/atspmv-mfem \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DATSPMV_BUILD_MFEM_APPS=ON -DMFEM_DIR=/tmp/mfem-build
cmake --build /tmp/atspmv-mfem --target mfem_heat --parallel 4
```

MFEM 与 AtSpMV 应使用相同 CUDA 工具链。普通构建默认不查找或下载 MFEM。

## 运行

```bash
CUDA_VISIBLE_DEVICES=3 /tmp/atspmv-mfem/apps/mfem/mfem_heat \
  -m /tmp/mfem/data/star.mesh -r 6 -o 2 \
  -s 21 -dt 0.01 -tf 1 -a 0 -b atbsr -sol /tmp/temperature.bin
```

`-s 21` 为隐式 Euler，`-s 23` 为原例的 SDIRK；前向 Euler 在大时间步下可能不稳定。
`-a 0` 是常系数热传导；`-a 0.01` 是温度相关的非线性情形。
常系数时各后端均复用不变的 K 和隐式矩阵，避免重复组装；非线性时均按原例
在时间步末更新 K。求解未收敛、温度非有限、热量守恒或能量衰减检查失败时返回非零退出码。
默认对完整 CSR 偏移和列索引进行精确比较：相同时通过 `updateSpmvValues` 更新数值，
复用已有行排序、shape 和槽位；结构改变时重建。该优化对 Plan cuSPARSE 也启用。
结构比较在 GPU 上逐项执行，只把一个判定标志传回 CPU，避免时间步内反复下载整份索引。
使用 `-rebuild` 可关闭复用以做消融实验。原生 `mfem` 后端直接使用 MFEM 自己的矩阵管理。
初始温度沿用上游的分片常数分布。
两种系数更新实现都会在访问前显式调用 `HostReadWrite()`。MFEM 的 `operator()(i)`
是原始主机访问；只启用 CUDA 而不处理有效性标志，会让 GPU 读取旧的扩散系数。
默认使用一次主机视图的批量系数循环；`-host-coeff` 选择逐元素循环作为等价对照。
默认同时复用隐式矩阵 `T=M+dt*K` 的符号并集：第一次调用构建 CSR 位置映射，
后续结构相同时在 GPU 更新 T 的数值；源矩阵结构变化则重建映射。使用 `-rebuild-t`
可关闭这一应用层优化。该优化对 MFEM 原生和全部 Plan 后端相同，须单独消融，
不能将其收益直接计为 tile shape 的收益。
为允许大网格收敛，隐式 CG 最大迭代次数统一设为 2000（上游例子为 100），
相对容差仍为 1e-8。所有后端使用同一设置和同一 Jacobi 预处理器。
`-fa` 可启用 MFEM 原有的 GPU full assembly；这是对所有后端共同的组装方式选择，
其收益不能归因于 AtSpMV。该路径须使用 MFEM 支持的元素和积分器组合。
`-persist-x` 可为当前 SpMV 输入配置 L2 保留窗口；设备支持范围和向量大小决定窗口与
命中比例，对所有后端使用相同逻辑。默认关闭，不使用 DSM；窗口按默认 stream 管理。
它不消除数值变化时对 X 的写入，只影响后续读取的缓存策略。

诊断时可用 `-dump-t /tmp/implicit.csr` 导出第一个隐式矩阵；这个过程增加主机拷贝和
文件输出，不应混入性能对照。文件依次为原生 int32 的 rows/cols/nnz、CSR 行偏移、
列索引及 FP64 数值。`cmake --build BUILD --target packed_spmv_bench` 构建独立重放工具，
运行 `BUILD/apps/mfem/packed_spmv_bench /tmp/implicit.csr [--persist-x]` 比较内核。

`APP_RESULT` 输出包含初始化、时间推进、Plan 构造、CG 迭代和矩阵乘调用数。
`total_s` 从 main 开始计时到时间推进结束，包含初始化、CUDA context、网格读取/细化、
有限元组装和全部 Plan 构造；不包含最终验证、温度场输出和析构。
`evolution_s` 包含时间步中的线性求解，以及该阶段发生的矩阵更新与 Plan 构造。
它不是纯 SpMV 时间。首次热量/能量检查在初始化计时内，最终检查在计时外。
`linear_s` 为 RHS 矩阵乘和 CG 的累计墙钟；`parameter_s` 为系数、K 更新及其 Plan
维护时间（包括初始化）；`implicit_setup_s` 为各次隐式矩阵构造和 Plan 维护时间。
这些计数用于定位瓶颈，不是三项可直接相加得到 `total_s` 的互斥区间。

批量验证使用 `scripts/analysis/benchmark_mfem_heat.py`；正序/逆序运行后端，
比较整个最终温度向量，同时保留迭代次数。字段临时写到本地 `/tmp`，后续归档时间
不计入子进程计时，以避免网络文件系统掩盖计算性能。不能把排除输出的 `total_s`
称为包含文件输出的完整进程时间；另记录的 `process_wall_s` 包含本地结果输出和析构。

该应用暂未验证非一致自适应网格的自由度约束和多 GPU 执行。大规模实验结果以
`results/applications/mfem-heat/` 中的原始日志和汇总为准。
独立验证使用上游 CPU ex16；运行 `scripts/analysis/validate_mfem_cpu_reference.py`
可比较两种细化程度和四种 GPU 后端的完整温度场。旧的未同步系数批次被标记为无效，
性能报告仅使用 `coherent-*` 系列。

## 已验证结果

H100 PCIe 上的校正版本已完成两组四轮对照。1313281 个自由度、20981761 个存储条目、
20 个隐式时间步的 AtBSR 方案相对每步重建参考为 1.617×；应用计算时间中位数由
15.512 s 降至 9.590 s。收益主要来自结构复用；相同优化下 cuSPARSE 为 9.366 s、
MFEM 原生为 9.378 s，因此该结果不证明 AtBSR 内核在此矩阵上更快。
全部 48 个应用进程通过完整温度场对照；另有上游 CPU 参考、54 个数值更新配置、
956 项 Plan 直接/Graph 检查及稀疏矩阵和缓存的内存检查。
详情和图表见 [应用报告](../../docs/mfem-application.md)。

后续压缩索引与 L2 窗口优化已经在同一大组完成独立八轮复测：压缩 AtBSR 的应用计算
时间中位数为 9.125 s，启用相同缓存和结构复用的 cuSPARSE 为 9.404 s，配对几何平均
加速比 1.0294x，八轮均获益。该结论限于这一 case；见应用报告的后续优化部分。
后续 [2x2 消融](../../docs/mfem-column-ablation.md) 显示：单独压缩在大矩阵上提高
SpMV/CG 性能，但未证明应用总时间有稳定额外收益，小矩阵会退化，默认仍关闭。

```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/analysis/analyze_mfem_application.py
```
