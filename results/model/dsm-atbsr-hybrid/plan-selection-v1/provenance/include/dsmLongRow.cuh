// Async reduction adapted from Quack (Copyright (c) 2025, Tri Dao),
// Apache-2.0: see third_party/quack/LICENSE and NOTICE.
// Ported from CuTe-DSL to FP64 CUDA/PTX with CSR row partitioning, short-row
// fusion, and a pull-based reference path. No Quack runtime dependency.
#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdexcept>
#include <climits>

namespace atspmv {
namespace long_row_detail {
// Every block first reduces its chunk locally. DSM transports one FP64
// partials, never x. A cluster owns Rows disjoint CSR rows.
template<int Mode, int Rows = 1>
__device__ __forceinline__ void reduceLongRows(const int* ptr, const int* col, const double* val,
                           const double* x, double* y, double* scratch,
                           int parts, const int* row_ids, int row_count) {
    namespace cg = cooperative_groups;
    __shared__ double sums[Rows][32];
    __shared__ unsigned long long transaction;
    __shared__ double inbox[32 * 8];
    unsigned barrier_address = 0;
    if constexpr (Mode == 4) {
        barrier_address = static_cast<unsigned>(__cvta_generic_to_shared(&transaction));
        if (threadIdx.x == 0)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(barrier_address) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    }
    const int threads_per_row = blockDim.x / Rows;
    const int local_row = threadIdx.x / threads_per_row;
    const int task = blockIdx.x / parts * Rows + local_row;
    const bool valid = task < row_count;
    const int row = valid ? (row_ids ? row_ids[task] : task) : 0;
    const int part = blockIdx.x % parts;
    const int tid = threadIdx.x % threads_per_row;
    const int lane = tid & 31, warp = tid / 32;
    const int begin = valid ? ptr[row] : 0;
    const int length = valid ? ptr[row + 1] - begin : 0;
    const int first = begin + (long long)length * part / parts;
    const int last = begin + (long long)length * (part + 1) / parts;
    double sum = 0;
    for (int j = first + tid; j < last; j += threads_per_row)
        sum += val[j] * x[col[j]];
    if constexpr (Mode == 4) {
        // Broadcast each warp partial to all cluster peers, as in Quack's
        // cluster_reduce. Byte-counted completion replaces remote reads and
        // an exit cluster barrier. Each CTA waits for ALL its incoming stores.
        for (int d = 16; d; d /= 2) sum += __shfl_xor_sync(0xffffffff, sum, d);
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        const unsigned bytes = (blockDim.x / 32) * parts * sizeof(double);
        if (threadIdx.x == 0)
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                         :: "r"(barrier_address), "r"(bytes) : "memory");
        if (lane < parts) {
            unsigned destination = static_cast<unsigned>(__cvta_generic_to_shared(&inbox[(threadIdx.x / 32) * parts + part]));
            unsigned remote_data, remote_barrier;
            asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_data) : "r"(destination), "r"(lane));
            asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_barrier) : "r"(barrier_address), "r"(lane));
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b64 [%0], %1, [%2];"
                         :: "r"(remote_data), "l"(__double_as_longlong(sum)), "r"(remote_barrier) : "memory");
        }
        unsigned ready;
        do {
            asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                         : "=r"(ready) : "r"(barrier_address) : "memory");
        } while (!ready);
        if (part == 0 && warp == 0) {
            double total = 0;
            const int count = threads_per_row / 32 * parts;
            for (int j = lane; j < count; j += 32) total += inbox[local_row * count + j];
            for (int d = 16; d; d /= 2) total += __shfl_down_sync(0xffffffff, total, d);
            if (lane == 0 && valid) y[row] = total;
        }
    } else {
    for (int d = 16; d; d /= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
    if (lane == 0) sums[local_row][warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < threads_per_row / 32 ? sums[local_row][lane] : 0;
        for (int d = 16; d; d /= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
        if (lane == 0) {
            if constexpr (Mode == 0) { if(valid) y[row] = sum; }
            if constexpr (Mode == 1) { if(valid) atomicAdd(y + row, sum); }
            if constexpr (Mode == 2) { if(valid) scratch[task * parts + part] = sum; }
            if constexpr (Mode == 3) sums[local_row][0] = sum;
        }
    }
    if constexpr (Mode == 3) {
        auto cluster = cg::this_cluster();
        cluster.sync();
        if (part == 0 && warp == 0) {
            double total = lane < parts ? *cluster.map_shared_rank(&sums[local_row][0], lane) : 0;
            for (int d = 16; d; d /= 2) total += __shfl_down_sync(0xffffffff, total, d);
            if (lane == 0 && valid) y[row] = total;
        }
        cluster.sync();
    }
    }
}
template<int Mode, int Rows = 1>
__global__ void reduceRows(const int* ptr, const int* col, const double* val,
                           const double* x, double* y, double* scratch,
                           int parts, const int* row_ids, int row_count,
                           const int* short_ids, int short_count) {
    const int long_blocks = ((row_count + Rows - 1) / Rows) * parts;
    if (blockIdx.x >= long_blocks) {
        // Whole clusters enter this branch. Short rows need no DSM or barrier.
        const int task = (blockIdx.x - long_blocks) * (blockDim.x / 32) + threadIdx.x / 32;
        const int lane = threadIdx.x & 31;
        double sum = 0;
        const int row = task < short_count ? short_ids[task] : 0;
        if (task < short_count)
            for (int j = ptr[row] + lane; j < ptr[row+1]; j += 32) sum += val[j] * x[col[j]];
        for (int d = 16; d; d /= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
        if (task < short_count && lane == 0) y[row] = sum;
        return;
    }

    reduceLongRows<Mode, Rows>(ptr, col, val, x, y, scratch, parts, row_ids, row_count);
}

// One cluster works exclusively on either DSM long rows or AtBSR short-row
// tiles. The packed recorder is remapped to original rows at plan creation.
template<int Rows, int TileM, typename Index>
__global__ void fusedRows(const int* ptr, const int* col, const double* val,
    const double* x, double* y, int parts, const int* row_ids, int row_count,
    const int* short_ids, int short_count, const int* block_ptr, const Index* block_idx,
    const double* block_val, const int* recorder, int packed_warps) {
    const int long_blocks = ((row_count + Rows - 1) / Rows) * parts;
    if (blockIdx.x < long_blocks) {
        reduceLongRows<4, Rows>(ptr, col, val, x, y, nullptr, parts, row_ids, row_count);
        return;
    }
    const int tid = (blockIdx.x-long_blocks)*blockDim.x+threadIdx.x;
    const int warp = tid/32, lane=threadIdx.x%32;
    if (!block_ptr) {
        if (tid < short_count) y[short_ids[tid]]=0.0;
        return;
    }
    if(warp>=packed_warps)return;
    constexpr int K=32/TileM;
    double sum=0;
    for(int b=block_ptr[warp];b<block_ptr[warp+1];++b) {
        int idx=static_cast<int>(block_idx[b*32+lane]);
        if(idx>=0)sum+=block_val[b*32+lane]*x[idx];
    }
    for(int d=1;d<K;d*=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
    const int row=warp*TileM+lane/K;
    if(lane%K==0 && row<short_count)y[recorder[row]]=sum;
}

__global__ void finish(const double* scratch, double* y, int rows, int parts, const int* row_ids) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    double sum = row < rows && lane < parts ? scratch[row * parts + lane] : 0;
    for (int d = 16; d; d /= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
    if (row < rows && lane == 0) y[row_ids ? row_ids[row] : row] = sum;
}
}
// Experimental FP64 CSR beta=0 launch; no allocations or output clear.
// row_ids optionally selects disjoint long rows (device array of row_count).
// short_ids selects optional short rows, handled by one warp per row in the
// SAME kernel. Lists must contain unique, in-range, mutually disjoint row IDs.
// Unlisted outputs are untouched. Arrays/lists must remain live; x/y must not
// overlap. Only beta=0 is supported. Requires SM90 and compilation for SM90.
// Rows=1/4/8 trades row parallelism against per-row threads. Async=false uses
// two full cluster barriers and remote loads as an experimental control.
// There is no automatic tuning or change to the production SpMV dispatcher.
template<int Rows = 1, bool Async = true>
inline void launchDsmLongRows(const int* ptr, const int* col, const double* val,
                              const double* x, double* y, int row_count,
                              int cluster_blocks = 4, int threads = 256,
                              cudaStream_t stream = nullptr, const int* row_ids = nullptr,
                              const int* short_ids = nullptr, int short_count = 0) {
    static_assert(Rows == 1 || Rows == 4 || Rows == 8);
    if (short_count < 0 || (short_count && !short_ids) || threads / Rows < 32 || row_count < 0 || (cluster_blocks != 2 && cluster_blocks != 4 && cluster_blocks != 8)
        || (threads != 128 && threads != 256 && threads != 512 && threads != 1024))
        throw std::invalid_argument("invalid DSM long-row launch configuration");
    const auto long_blocks = ((static_cast<long long>(row_count) + Rows - 1) / Rows) * cluster_blocks;
    const auto short_blocks = ((static_cast<long long>(short_count) + (threads / 32) * cluster_blocks - 1) / ((threads / 32) * cluster_blocks)) * cluster_blocks;
    if (long_blocks + short_blocks > INT_MAX || row_count > INT_MAX - Rows || short_count > INT_MAX - threads)
        throw std::invalid_argument("DSM long-row grid exceeds 32-bit indexing");
    if (!row_count && !short_count) return;
    if (!ptr || !col || !val || !x || !y || x == y)
        throw std::invalid_argument("invalid DSM long-row arrays");
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {static_cast<unsigned>(cluster_blocks), 1, 1};
    cudaLaunchConfig_t config{};
    const unsigned short_clusters = (static_cast<unsigned>(short_count) + (threads / 32) * cluster_blocks - 1) / ((threads / 32) * cluster_blocks);
    config.gridDim = dim3((((static_cast<unsigned>(row_count) + Rows - 1) / Rows) + short_clusters) * cluster_blocks);
    config.blockDim = dim3(threads);
    config.stream = stream;
    config.attrs = &attr; config.numAttrs = 1;
    auto error = cudaLaunchKernelEx(&config, long_row_detail::reduceRows<Async ? 4 : 3, Rows>,
                                  ptr, col, val, x, y, static_cast<double*>(nullptr), cluster_blocks, row_ids, row_count, short_ids, short_count);
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace atspmv
