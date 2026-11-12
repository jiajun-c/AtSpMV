#pragma once

#include <cuda_runtime_api.h>
#include "common.h"
#include "matrixFormat.h"
#include "tileModel.h"

// Scratch local to one unchanged sorted CSR row order. Invalidated when the
// task tile capacity changes; never shared between different matrices.
struct AtBSRProductStatsCache {
    atspmv::TileStats stats[6]{};
    const int* sorted_lengths = nullptr;
    int rows = 0;
    int tiles_per_task = 0;
    bool initialized = false;
};

int CSRToAtBSRProduct(CSRMatrix<double, int32_t>&csrMat, AtBsr<double, int32_t>&atbsr, int idx, int balance = -1);
// Precompute the sorted row lengths and original row indices in caller-owned
// device arrays. They must match this CSR and remain valid during packing.
void prepareAtBSRProductRowOrder(const CSRMatrix<double, int32_t>& csr, int* lengths, int* indices);
int CSRToAtBSRProductFromRows(CSRMatrix<double, int32_t>& csr, AtBsr<double, int32_t>& atbsr,
                             int idx, int balance, const int* lengths, const int* indices,
                             AtBSRProductStatsCache* stats_cache = nullptr);
int AtSpmvProduct(AtBsr<double, int32_t>& atbsr, double* d_x, double* d_y, double alpha, double beta, cudaStream_t stream = nullptr);
