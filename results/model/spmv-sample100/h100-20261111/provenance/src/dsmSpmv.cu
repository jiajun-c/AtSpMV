#include "dsmSpmv.h"
#include "tileSelector.cuh"
#include <cooperative_groups.h>
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace atspmv {
namespace {

template<class Index, int K, bool Balanced, int ClusterBlocks, bool StageX>
__global__ void dsmSpmv(const int* targets, const int* ptr, const Index* columns,
                        const double* values, int tasks, int rows, int cols,
                        const double* x, double* y, const int* permutation) {
    extern __shared__ double shared_x[];
    auto cluster = cooperative_groups::this_cluster();
    if constexpr (StageX) {
        // Cyclic shards make each gather's owner/offset a mask and a shift.
        // ClusterBlocks is a compile-time power of two, including 1.
        for (int i = threadIdx.x; i < (cols + ClusterBlocks - 1) / ClusterBlocks;
             i += blockDim.x) {
            const int col = i * ClusterBlocks + cluster.block_rank();
            shared_x[i] = col < cols ? x[col] : 0.0;
        }
        // Publishes each shard and ensures all remote CTA shared memory exists.
        cluster.sync();
    }
    const int lane = threadIdx.x & 31;
    // CTA b / warp w owns tasks b*W+w+k*(gridDim.x*W). In normal mode,
    // task t owns reordered rows [t*(32/K), (t+1)*(32/K)); permutation maps
    // them to original rows. Balanced tasks use targets[t] and can split a
    // row group across tasks/CTAs, hence atomicAdd below. Not one row/CTA.
    const int stride = gridDim.x * (blockDim.x / 32);
    for (int task = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
         task < tasks; task += stride) {
        double sum = 0;
        for (int tile = ptr[task]; tile < ptr[task + 1]; ++tile) {
            const int offset = tile * 32 + lane;
            const int col = int(columns[offset]);
            if (col >= 0) {
                double value;
                if constexpr (StageX) {
                    const int owner = col & (ClusterBlocks - 1);
                    const double* shard = cluster.map_shared_rank(shared_x, owner);
                    value = shard[col / ClusterBlocks];
                } else {
                    value = x[col];
                }
                sum += values[offset] * value;
            }
        }
        #pragma unroll
        for (int offset = 1; offset < K; offset *= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset, K);
        if (lane % K == 0) {
            const int row = (Balanced ? targets[task] : task * (32 / K)) + lane / K;
            if (row < rows) {
                if constexpr (Balanced) atomicAdd(y + permutation[row], sum);
                else y[permutation[row]] = sum;
            }
        }
    }
    // Even CTAs with no assigned task must keep their x shard alive until
    // every remote reader finishes. No early return is allowed above this.
    if constexpr (StageX) cluster.sync();
}

template<class Index, int K, bool Balanced, int N>
const void* storageKernel(XStorage storage) {
    if (storage == XStorage::DistributedShared)
        return reinterpret_cast<const void*>(dsmSpmv<Index, K, Balanced, N, true>);
    return reinterpret_cast<const void*>(dsmSpmv<Index, K, Balanced, N, false>);
}
template<class Index, int K, bool Balanced>
const void* clusterKernel(int blocks, XStorage storage) {
    switch (blocks) {
    case 1: return storageKernel<Index, K, Balanced, 1>(storage);
    case 2: return storageKernel<Index, K, Balanced, 2>(storage);
    case 4: return storageKernel<Index, K, Balanced, 4>(storage);
    case 8: return storageKernel<Index, K, Balanced, 8>(storage);
    default: throw std::invalid_argument("DSM cluster size must be 1, 2, 4 or 8");
    }
}
template<class Index, int K>
const void* modeKernel(bool balanced, int blocks, XStorage storage) {
    return balanced ? clusterKernel<Index, K, true>(blocks, storage)
                    : clusterKernel<Index, K, false>(blocks, storage);
}
template<class Index>
const void* shapeKernel(const AtBsr<double, int32_t>& a, int blocks, XStorage storage) {
    switch (a.TILE_K) {
    case 32: return modeKernel<Index, 32>(a.useBalance, blocks, storage);
    case 16: return modeKernel<Index, 16>(a.useBalance, blocks, storage);
    case 8: return modeKernel<Index, 8>(a.useBalance, blocks, storage);
    case 4: return modeKernel<Index, 4>(a.useBalance, blocks, storage);
    case 2: return modeKernel<Index, 2>(a.useBalance, blocks, storage);
    case 1: return modeKernel<Index, 1>(a.useBalance, blocks, storage);
    default: throw std::invalid_argument("invalid DSM tile shape");
    }
}

cudaLaunchConfig_t launchConfig(int blocks, int threads, std::size_t shared,
                                int cluster_blocks, cudaLaunchAttribute& attr,
                                cudaStream_t stream = nullptr) {
    attr = {};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {static_cast<unsigned>(cluster_blocks), 1, 1};
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(blocks);
    config.blockDim = dim3(threads);
    config.dynamicSmemBytes = shared;
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;
    return config;
}
} // namespace

DsmSpmvPlan makeDsmSpmvPlan(const AtBsr<double, int32_t>& a, int cluster_blocks,
                          XStorage storage, int max_clusters, bool single_block_per_sm, int warps_per_block) {
    if ((cluster_blocks != 1 && cluster_blocks != 2 && cluster_blocks != 4 && cluster_blocks != 8) ||
        max_clusters < 0 || a.ncols < 0 || a.ncols > INT_MAX - 8 || a.nrows < 0 || a.nnz < 0 ||
        (storage != XStorage::Global && storage != XStorage::DistributedShared))
        throw std::invalid_argument("invalid DSM plan configuration");
    if (warps_per_block != 0 && warps_per_block != 2 && warps_per_block != 4 &&
        warps_per_block != 8 && warps_per_block != 16 && warps_per_block != 32)
        throw std::invalid_argument("invalid DSM warps per block");
    a.kernelConfig.validate();
    if (a.kernelConfig.prefetch || a.kernelConfig.cta_reduce)
        throw std::invalid_argument("DSM experiment requires prefetch=0 and cta_reduce=0");
    DsmSpmvPlan plan;
    modelCudaCheck(cudaGetDevice(&plan.device_));
    int supported = 0;
    modelCudaCheck(cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, plan.device_));
    if (!supported) throw std::runtime_error("DSM experiment requires a cluster-capable SM90+ GPU");
    plan.matrix_ = &a;
    plan.cluster_blocks_ = cluster_blocks;
    plan.threads_ = (warps_per_block ? warps_per_block : a.kernelConfig.warps_per_block) * 32;
    plan.storage_ = storage;
    // Empty inputs have no vector reads and only require zeroing output.
    if (a.nnz == 0 || a.nrows == 0) return plan;
    plan.kernel_ = a.useColIdxInt16 ? shapeKernel<int16_t>(a, cluster_blocks, storage)
                                  : shapeKernel<int32_t>(a, cluster_blocks, storage);
    if (storage == XStorage::DistributedShared)
        plan.shared_bytes_ = ((std::size_t(a.ncols) + cluster_blocks - 1) / cluster_blocks) * sizeof(double);
    if (single_block_per_sm) {
        int per_sm = 0;
        modelCudaCheck(cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, plan.device_));
        // Reserve the same capacity in global and DSM controls. Even if a
        // smaller carveout is chosen, two such CTAs cannot share one SM.
        plan.shared_bytes_ = std::max(plan.shared_bytes_, std::size_t(per_sm / 2 + 1024));
    }
    int limit = 0;
    modelCudaCheck(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, plan.device_));
    cudaFuncAttributes attributes{};
    modelCudaCheck(cudaFuncGetAttributes(&attributes, plan.kernel_));
    if (plan.threads_ > attributes.maxThreadsPerBlock)
        throw std::invalid_argument("DSM block configuration cannot be scheduled");
    if (plan.shared_bytes_ + attributes.sharedSizeBytes > std::size_t(limit))
        throw std::invalid_argument("entire x does not fit in this cluster's shared memory");
    // Opt in to the maximum, not the current vector size: plans for a larger x
    // must remain launchable after a smaller plan for the same kernel is made.
    if (plan.shared_bytes_)
        modelCudaCheck(cudaFuncSetAttribute(plan.kernel_, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           limit - int(attributes.sharedSizeBytes)));
    if (a.warpCount <= 0) throw std::invalid_argument("nonempty DSM matrix has no tasks");
    const int needed = int((static_cast<long long>(a.warpCount) +
                            (plan.threads_ / 32) * cluster_blocks - 1) /
                           ((plan.threads_ / 32) * cluster_blocks));
    cudaLaunchAttribute attr{};
    auto config = launchConfig(std::max(1, needed) * cluster_blocks, plan.threads_,
                               plan.shared_bytes_, cluster_blocks, attr);
    modelCudaCheck(cudaOccupancyMaxActiveClusters(&plan.active_clusters_, plan.kernel_, &config));
    if (plan.active_clusters_ <= 0)
        throw std::invalid_argument("DSM block configuration cannot be scheduled");
    int clusters = std::min(needed, plan.active_clusters_);
    if (max_clusters) clusters = std::min(clusters, max_clusters);
    plan.grid_blocks_ = std::max(1, clusters) * cluster_blocks;
    return plan;
}

void launchDsmSpmv(const DsmSpmvPlan& plan, const double* x, double* y, cudaStream_t stream) {
    if (!plan.matrix_) throw std::invalid_argument("uninitialized DSM plan");
    int device;
    modelCudaCheck(cudaGetDevice(&device));
    if (device != plan.device_) throw std::invalid_argument("DSM plan used on a different CUDA device");
    const auto& a = *plan.matrix_;
    if (a.nrows == 0) return;
    if (!y || (a.nnz && !x) || x == y) throw std::invalid_argument("invalid or aliased DSM vectors");
    if (a.nnz == 0 || a.useBalance)
        modelCudaCheck(cudaMemsetAsync(y, 0, std::size_t(a.nrows) * sizeof(double), stream));
    if (a.nnz == 0) return;
    cudaLaunchAttribute attr{};
    auto config = launchConfig(plan.grid_blocks_, plan.threads_, plan.shared_bytes_, plan.cluster_blocks_, attr, stream);
    const int* ptr = a.useBalance ? a.warpBlcPtr : a.blcPtr;
    const void* columns = a.useColIdxInt16 ? static_cast<const void*>(a.blcIdxInt16) : a.blcIdx;
    const int* targets = a.targetRowByWarp;
    const double* values = a.blcVal;
    const int* permutation = a.recorder;
    int tasks = a.warpCount, rows = a.nrows, cols = a.ncols;
    void* args[] = {&targets, &ptr, &columns, &values, &tasks, &rows, &cols, &x, &y, &permutation};
    modelCudaCheck(cudaLaunchKernelExC(&config, plan.kernel_, args));
}
} // namespace atspmv
