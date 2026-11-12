#pragma once

#include "tileSelector.cuh"

namespace atspmv {
// A depth-sized batch exposes independent column gathers and accumulation chains.
// All lanes participate in shuffles, including padded rows and tail batches.
template<class Index, int K, bool Balanced, int Depth, bool Merge>
__global__ void tunedSpmv(const int* targets, const int* ptr, const Index* columns,
                          const double* values, int tasks, int rows,
                          const double* x, double* y, const int* permutation) {
    const int lane = threadIdx.x & 31;
    const int local_warp = threadIdx.x >> 5;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const bool valid = warp < tasks;
    // Merge needs every CTA thread at the barrier, even in the last partial CTA.
    if constexpr (!Merge) { if (!valid) return; }
    const int row_base = valid ? (Balanced ? targets[warp] : warp * (32 / K)) : -1;
    const int begin = valid ? ptr[warp] : 0;
    const int end = valid ? ptr[warp + 1] : 0;
    double accum[Depth] = {};
    for (int tile = begin; tile < end; tile += Depth) {
        int col[Depth];
        double a[Depth], b[Depth];
        #pragma unroll
        for (int j = 0; j < Depth; ++j) {
            const bool active = tile + j < end;
            col[j] = active ? int(columns[(tile + j) * 32 + lane]) : -1;
            a[j] = active && col[j] >= 0 ? values[(tile + j) * 32 + lane] : 0.0;
        }
        #pragma unroll
        for (int j = 0; j < Depth; ++j) b[j] = col[j] >= 0 ? x[col[j]] : 0.0;
        #pragma unroll
        for (int j = 0; j < Depth; ++j) accum[j] += a[j] * b[j];
    }
    double sum = 0;
    #pragma unroll
    for (int j = 0; j < Depth; ++j) sum += accum[j];
    #pragma unroll
    for (int offset = 1; offset < K; offset *= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset, K);

    if constexpr (Merge) {
        static_assert(Balanced, "CTA merging is only needed for split tasks");
        __shared__ int group[8];
        __shared__ double partial[8][32 / K];
        if (lane == 0) group[local_warp] = row_base;
        if (lane % K == 0) partial[local_warp][lane / K] = sum;
        __syncthreads();
        // Tasks of a row group are consecutive. The first warp of each run
        // owns its CTA-local sum. A group crossing CTAs still uses atomicAdd.
        if (valid && lane % K == 0 &&
            (local_warp == 0 || group[local_warp - 1] != row_base)) {
            for (int w = local_warp + 1; w < blockDim.x / 32 && group[w] == row_base; ++w)
                sum += partial[w][lane / K];
            const int row = row_base + lane / K;
            if (row < rows) atomicAdd(y + permutation[row], sum);
        }
    } else if (lane % K == 0) {
        const int row = row_base + lane / K;
        if (row < rows) {
            if constexpr (Balanced) atomicAdd(y + permutation[row], sum);
            else y[permutation[row]] = sum;
        }
    }
}

template<class Index, int K, int Depth>
inline const void* tunedMode(bool balanced, bool merge) {
    if (!balanced) return reinterpret_cast<const void*>(tunedSpmv<Index, K, false, Depth, false>);
    if (merge) return reinterpret_cast<const void*>(tunedSpmv<Index, K, true, Depth, true>);
    return reinterpret_cast<const void*>(tunedSpmv<Index, K, true, Depth, false>);
}
template<class Index, int K>
inline const void* tunedDepth(bool balanced, const KernelConfig& c) {
    switch (c.prefetch) {
    case 2: return tunedMode<Index, K, 2>(balanced, c.cta_reduce);
    case 4: return tunedMode<Index, K, 4>(balanced, c.cta_reduce);
    case 8: return tunedMode<Index, K, 8>(balanced, c.cta_reduce);
    default: return tunedMode<Index, K, 1>(balanced, c.cta_reduce);
    }
}
template<class Index>
inline const void* tunedKernel(int index, bool balanced, const KernelConfig& c) {
    switch (index) {
    case 0: return tunedDepth<Index, 32>(balanced, c);
    case 1: return tunedDepth<Index, 16>(balanced, c);
    case 2: return tunedDepth<Index, 8>(balanced, c);
    case 3: return tunedDepth<Index, 4>(balanced, c);
    case 4: return tunedDepth<Index, 2>(balanced, c);
    case 5: return tunedDepth<Index, 1>(balanced, c);
    default: throw std::invalid_argument("invalid tuned tile index");
    }
}
inline const void* tunedKernel(int index, bool balanced, bool small, const KernelConfig& c) {
    return small ? tunedKernel<int16_t>(index, balanced, c) : tunedKernel<int32_t>(index, balanced, c);
}
} // namespace atspmv
