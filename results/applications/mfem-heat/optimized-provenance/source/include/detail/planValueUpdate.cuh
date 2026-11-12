#pragma once
#include "mixedTile.cuh"

namespace atspmv::detail {
// The sorted row order and tile offsets depend only on the CSR structure.
// Both normal and balanced AtBSR use the same packed value layout.
static __global__ void updateAtBsrValues(const int* ptr, const int* order,
    const int* tiles, int rows, int m, int k, const double* values, double* packed) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp >= (rows + m - 1) / m) return;
    int item = warp * m + lane / k;
    int row = item < rows ? order[item] : -1;
    for (int tile = tiles[warp]; tile < tiles[warp+1]; ++tile) {
        int pos = (tile - tiles[warp]) * k + lane % k;
        packed[size_t(tile) * 32 + lane] = row >= 0 && pos < ptr[row+1]-ptr[row]
            ? values[ptr[row] + pos] : 0.;
    }
}
static __global__ void updateMixedValues(const Group* groups, int count,
    const int* order, const int* ptr, int rows, const double* values, double* packed) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp >= count) return;
    Group group = groups[warp];
    int k = 32 / group.m, item = group.row_start + lane / k;
    int row = item < rows ? order[item] : -1;
    for (int tile = group.tile_start; tile < group.tile_end; ++tile) {
        int pos = (tile - group.tile_start) * k + lane % k;
        packed[size_t(tile) * 32 + lane] = row >= 0 && pos < ptr[row+1]-ptr[row]
            ? values[ptr[row] + pos] : 0.;
    }
}
} // namespace atspmv::detail
