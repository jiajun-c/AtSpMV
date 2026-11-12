# CUDA Core 与 Tensor Core SpMM 对照

H100 PCIe 3 号卡在后台任务启动时有其他进程间歇占用；运行器等待连续空闲，运行中检测到外部计算进程后中断并重试。现有 36 个案例（18 组矩阵及两种构建顺序）均完成、正确性检查通过。[对照数据](../results/model/cusparse-spmm-core-compare/h100-synthetic-v1/comparison.csv) 与 [汇总](../results/model/cusparse-spmm-core-compare/h100-synthetic-v1/summary.json)。后续 Nsight Compute 采样已完成四组配置，对每个 case 分别采样三种算法；前处理与计算 kernel 分开判读，不能把采样结果推广到未采样输入。

按两种顺序各自时间比的几何平均比较，18 组中 CSR ALG3 有 12 组快于 Blocked-ELL ALG1 超过 5%，4 组慢超过 5%，2 组在 5% 范围内。CSR ALG2 在这 18 组中均慢超过 5%；不能将此结论推广到其他矩阵或精度。最稀疏的一组（每块行 2 块、块内实际密度 4.57%、N=32），ALG3 为 2.47 μs，Blocked-ELL 为 3.81 μs，后者/前者为 1.54×；同样块行配置下密度 93.23%、N=32 时分别为 4.44 μs 和 3.80 μs。不同格式与算法同时变化，这些结果不能单独证明硬件计算单元造成差异。

**指令核验修正：CSR ALG3 并不总是纯 CUDA Core。** 在每块行 2 块、密度 4.57%、N=32 的采样中，其 `csrmm_v3_kernel` 的 `sm__inst_executed_pipe_tensor_op_hmma.sum` 为 792；Blocked-ELL 的 `EllGemm` 为 4096。上面的 1.54× 是两条均含 HMMA 的路径对比，不能称作 CUDA Core 快过 Tensor Core。在每块行 8 块、密度约 4.61%、N=8 的采样中，CSR ALG3 的计算 kernel Tensor/HMMA 计数均为 0、FFMA 线程指令计数为 523264；Blocked-ELL 计算 kernel HMMA 计数为 16384。对应无 profiler 的计时为 2.57 μs 对 5.06 μs，约 1.97×，是该具体输入上未观察到 Tensor 指令的 CSR 路径快于包含 HMMA 的 ELL 路径的证据。计数针对单个采样 kernel 的一次 launch；NCU 采样的时间不能作为性能比较。不同格式和调度仍然同时变化，因此不能把差距全部归因于计算单元。[原始计数](../results/model/cusparse-spmm-core-compare/h100-synthetic-v1/instruction-profile/)。

已有论文提供存在性证据：[Voltrix-SpMM](https://www.usenix.org/system/files/atc25-xia.pdf) 在 H100 PCIe 图数据实验中报告，其 CUDA Core 的 RoDe 对照快于 Tensor Core 的 TC-GNN 和 DTC-SpMM 对照；其图数据实验使用 N=256、512、1024，实验条件与本库不同。该论文同样报告其新的 Tensor Core 方法整体快于 RoDe，因此不能据此断言 CUDA Core 普遍胜出。应逐项核对实现的精度、数据、统计口径和预处理是否计入，不能直接套用到 FP64。

## 本机实验设计

`bench/cusparse_spmm_core_compare.cu` 对同一张合成稀疏矩阵生成 CSR 与 16×16 Blocked-ELL，执行 `C=A*B`：A、B 使用 FP16 输入，C 和计算类型 FP32，B/C 行主序。32-bit 索引；每种矩阵都在 CPU 从 CSR 构建参考结果，每个候选检查输出。矩阵为 1024×1024，非空块每块行 2 或 8 个，生成时指定块内 5%、25%、100% 的保留概率，实际非零率单独记录；N 为 8、32、128。两种反向候选构造顺序各 9 次计时，结果对两顺序取几何平均。预热 Graph 内事件时间不含格式转换、准备、预处理和 Graph 创建成本。

候选为 cuSPARSE CSR ALG2、ALG3 和 Blocked-ELL ALG1。后者是 NVIDIA 文档中支持 Tensor Core 的格式和精度，但是否真正执行 Tensor 指令取决于实现及输入；不能仅根据格式推断。Nsight Compute 的 `sm__inst_executed_pipe_tensor.sum` 与 `sm__inst_executed_pipe_tensor_op_hmma.sum` 已采样四组配置，不是每个计时案例的指令证明。前向/反向各个 case 的状态及错误会保留，缺失/错误不参与性能统计。

此实验主要回答这些**合成 case 是否存在一个较快的 CSR 路径**，不足以证明真实矩阵、FP64、其他 GPU、其他 N 或所有 SOTA 方法的关系。数值相同和比较过的候选分别占用不同格式，不单独隔离硬件管线与数据结构的影响；即使分别核验了 Tensor 指令，也只能说相应实现和格式的整体性能不同。

## 复现

```bash
cmake --build <build-dir> --target cusparse_spmm_core_compare
python3 tests/cusparse_spmm_core_analysis_test.py
```

跑批及冻结执行文件见结果目录；选择设备时确保无其他活跃作业。`scripts/analysis/run_cusparse_spmm_cores.sh` 使用 `ATSPMV_SPMM_GPU_INDEX` 指定设备，默认 3。原始结果、退出状态、对照表和 Nsight Compute 结果都保留在同一输出目录。
