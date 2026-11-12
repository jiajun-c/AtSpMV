#pragma once

#include "cudaError.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace atspmv::detail {

inline void mixedCheck(cudaError_t error) {
    if (error != cudaSuccess) throw CudaError(error);
}

template<class T>
T* mixedUpload(const std::vector<T>& values) {
    T* device = nullptr;
    mixedCheck(cudaMalloc(&device, std::max(size_t(1), values.size()) * sizeof(T)));
    try {
        if (!values.empty())
            mixedCheck(cudaMemcpy(device, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    } catch (...) {
        cudaFree(device);
        throw;
    }
    return device;
}

// Each group covers at most m adjacent rows in the length-sorted row order.
// A tile has m rows, 32/m slots per row, and 32 slots in total.
struct Group {
    int row_start;
    int tile_start;
    int tile_end;
    int m;
};

template<class Index>
__global__ void mixed_spmv(const Group* groups, int count, const int* row_ids, int nrows,
                           const Index* index, const double* value, const double* x, double* y) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= count) return;
    const Group group = groups[warp];
    const int width = 32 / group.m;
    double sum = 0;
    for (int tile = group.tile_start; tile < group.tile_end; ++tile) {
        const int slot = tile * 32 + lane;
        const int col = int(index[slot]);
        if (col >= 0) sum += value[slot] * x[col];
    }
    for (int offset = width / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset, width);
    const int row = group.row_start + lane / width;
    if (lane % width == 0 && row < nrows) y[row_ids[row]] = sum;
}

template<int M, bool Balanced, class Index>
__device__ __forceinline__ void mixed_fixed_group(Group group, int lane, const int* row_ids, int nrows,
                                                  const Index* index, const double* value,
                                                  const double* x, double* y) {
    constexpr int width = 32 / M;
    double sum = 0;
    for (int tile = group.tile_start; tile < group.tile_end; ++tile) {
        const int slot = tile * 32 + lane;
        const int col = int(index[slot]);
        if (col >= 0) sum += value[slot] * x[col];
    }
    for (int offset = width / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset, width);
    const int row = group.row_start + lane / width;
    if (lane % width == 0 && row < nrows) {
        if constexpr (Balanced) atomicAdd(y + row_ids[row], sum);
        else y[row_ids[row]] = sum;
    }
}

template<bool Balanced, class Index>
__global__ void mixed_spmv_specialized(const Group* groups, int count, const int* row_ids, int nrows,
                                       const Index* index, const double* value, const double* x, double* y) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= count) return;
    const Group group = groups[warp];
    switch (group.m) {
#define SHAPE(M) case M: mixed_fixed_group<M, Balanced>(group, lane, row_ids, nrows, index, value, x, y); break
        SHAPE(1); SHAPE(2); SHAPE(4); SHAPE(8); SHAPE(16); SHAPE(32);
#undef SHAPE
    }
}

// Owns the packed values, indices and the descriptors for all five modes.
struct Mixed {
    std::vector<int> order;
    std::vector<Group> groups;
    std::array<int, 6> histogram{};
    int tiles = 0;
    Group* dg = nullptr;
    Group* dtasks4 = nullptr;
    Group* dtasks8 = nullptr;
    Group* dtasks16 = nullptr;
    int ntasks4 = 0, ntasks8 = 0, ntasks16 = 0;
    int* drows = nullptr;
    void* didx = nullptr;
    double* dval = nullptr;
    bool small = false;
    size_t stored_bytes = 0, bytes4 = 0, bytes8 = 0, bytes16 = 0;

    Mixed() = default;
    Mixed(const Mixed&) = delete;
    Mixed& operator=(const Mixed&) = delete;
    ~Mixed() {
        for (void* p : {static_cast<void*>(dg), static_cast<void*>(dtasks4),
                        static_cast<void*>(dtasks8), static_cast<void*>(dtasks16),
                        static_cast<void*>(drows), didx, static_cast<void*>(dval)}) {
            if (p) cudaFree(p);
        }
    }
};

inline std::vector<Group> splitMixedTasks(const std::vector<Group>& groups, int cap) {
    std::vector<Group> tasks;
    for (const auto& group : groups) {
        for (int first = group.tile_start; first < group.tile_end; first += cap) {
            tasks.push_back({group.row_start, first, std::min(group.tile_end, first + cap), group.m});
        }
    }
    return tasks;
}

inline void make_mixed(Mixed& out, const std::vector<int>& ptr, const std::vector<int>& col,
                       const std::vector<double>& val, int ncols) {
    const int nrows = static_cast<int>(ptr.size()) - 1;
    out.order.resize(nrows);
    std::iota(out.order.begin(), out.order.end(), 0);
    std::stable_sort(out.order.begin(), out.order.end(), [&](int a, int b) {
        return ptr[a + 1] - ptr[a] < ptr[b + 1] - ptr[b];
    });

    // Find the least number of slots for adjacent groups in this row order.
    std::vector<long long> cost(size_t(nrows) + 1, 0);
    std::vector<unsigned char> shape(nrows);
    for (int i = nrows - 1; i >= 0; --i) {
        long long best = std::numeric_limits<long long>::max();
        int best_shape = 0;
        for (int t = 0; t < 6; ++t) {
            const int m = 1 << t, width = 32 / m, end = std::min(nrows, i + m);
            const int row = out.order[end - 1];
            const long long length = ptr[row + 1] - ptr[row];
            const long long tiles = (length + width - 1) / width;
            const long long total = tiles * 32 + cost[end];
            if (total <= best) { best = total; best_shape = t; }
        }
        cost[i] = best;
        shape[i] = static_cast<unsigned char>(best_shape);
    }
    if (cost[0] > INT32_MAX) throw std::overflow_error("mixed packed offsets exceed int32");

    for (int i = 0; i < nrows;) {
        const int t = shape[i], m = 1 << t, width = 32 / m, end = std::min(nrows, i + m);
        const int row = out.order[end - 1];
        const long long length = ptr[row + 1] - ptr[row];
        const int count = static_cast<int>((length + width - 1) / width);
        out.groups.push_back({i, out.tiles, out.tiles + count, m});
        ++out.histogram[t];
        out.tiles += count;
        i = end;
    }

    // Keep groups of the same shape and their data adjacent for the kernel.
    std::stable_sort(out.groups.begin(), out.groups.end(), [](const Group& a, const Group& b) {
        return a.m < b.m;
    });
    int next_tile = 0;
    for (auto& group : out.groups) {
        const int count = group.tile_end - group.tile_start;
        group.tile_start = next_tile;
        group.tile_end = next_tile + count;
        next_tile += count;
    }

    std::vector<double> packed(size_t(out.tiles) * 32, 0.0);
    std::vector<int32_t> indexes(packed.size(), -1);
    for (const auto& group : out.groups) {
        const int width = 32 / group.m;
        for (int row_offset = 0; row_offset < group.m && group.row_start + row_offset < nrows; ++row_offset) {
            const int row = out.order[group.row_start + row_offset];
            for (int j = ptr[row]; j < ptr[row + 1]; ++j) {
                const int pos = j - ptr[row];
                const size_t slot = size_t(group.tile_start + pos / width) * 32 + row_offset * width + pos % width;
                packed[slot] = val[j];
                indexes[slot] = col[j];
            }
        }
    }

    const auto tasks4 = splitMixedTasks(out.groups, 4);
    const auto tasks8 = splitMixedTasks(out.groups, 8);
    const auto tasks16 = splitMixedTasks(out.groups, 16);
    out.ntasks4 = tasks4.size(); out.ntasks8 = tasks8.size(); out.ntasks16 = tasks16.size();
    out.small = ncols < 32767;
    out.dg = mixedUpload(out.groups);
    out.dtasks4 = mixedUpload(tasks4); out.dtasks8 = mixedUpload(tasks8); out.dtasks16 = mixedUpload(tasks16);
    out.drows = mixedUpload(out.order);
    out.dval = mixedUpload(packed);
    if (out.small) {
        const std::vector<int16_t> small_indexes(indexes.begin(), indexes.end());
        out.didx = mixedUpload(small_indexes);
    } else {
        out.didx = mixedUpload(indexes);
    }
    out.stored_bytes = packed.size() * (sizeof(double) + (out.small ? sizeof(int16_t) : sizeof(int32_t))) +
                       out.groups.size() * sizeof(Group) + out.order.size() * sizeof(int);
    out.bytes4 = out.stored_bytes - out.groups.size() * sizeof(Group) + tasks4.size() * sizeof(Group);
    out.bytes8 = out.stored_bytes - out.groups.size() * sizeof(Group) + tasks8.size() * sizeof(Group);
    out.bytes16 = out.stored_bytes - out.groups.size() * sizeof(Group) + tasks16.size() * sizeof(Group);
}

template<class Index>
inline void launchMixedTyped(const Mixed& packed, int cap, int rows, const double* x, double* y,
                             cudaStream_t stream) {
    constexpr int threads = 128, warps_per_block = threads / 32;
    const auto* index = static_cast<const Index*>(packed.didx);
    if (cap <= 0) {
        const int count = static_cast<int>(packed.groups.size());
        if (!count) return;
        const int blocks = (count + warps_per_block - 1) / warps_per_block;
        if (cap == -1)
            mixed_spmv<Index><<<blocks, threads, 0, stream>>>(packed.dg, count, packed.drows, rows, index, packed.dval, x, y);
        else
            mixed_spmv_specialized<false, Index><<<blocks, threads, 0, stream>>>(packed.dg, count, packed.drows, rows, index, packed.dval, x, y);
    } else {
        const int count = cap == 4 ? packed.ntasks4 : cap == 8 ? packed.ntasks8 : packed.ntasks16;
        const Group* tasks = cap == 4 ? packed.dtasks4 : cap == 8 ? packed.dtasks8 : packed.dtasks16;
        mixedCheck(cudaMemsetAsync(y, 0, size_t(rows) * sizeof(double), stream));
        if (count) {
            const int blocks = (count + warps_per_block - 1) / warps_per_block;
            mixed_spmv_specialized<true, Index><<<blocks, threads, 0, stream>>>(tasks, count, packed.drows, rows, index, packed.dval, x, y);
        }
    }
    mixedCheck(cudaGetLastError());
}

inline void launchMixed(const Mixed& packed, int cap, int rows, const double* x, double* y, cudaStream_t stream) {
    if (packed.small) launchMixedTyped<int16_t>(packed, cap, rows, x, y, stream);
    else launchMixedTyped<int32_t>(packed, cap, rows, x, y, stream);
}

} // namespace atspmv::detail
