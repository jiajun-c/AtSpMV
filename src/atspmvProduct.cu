#include "detail/rowLengths.cuh"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include "atspmvprod.h"
#include "common.h"
#include "ptx.h"
#include "tileSelector.cuh"
#include "tunedSpmv.cuh"

static int configuredResidentWarps(int index, bool balanced, bool int16, const atspmv::KernelConfig& config);

using atspmv::detail::RowNNZCalculator;

__global__ void getBlcPtrKernelProduct(const int *drow_len,int tile_m,int tile_k, int nrows, int* blc_ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (nrows + tile_m - 1)/tile_m) return;
    int targetRow = min(nrows-1, (idx+1)*tile_m - 1);
    blc_ptr[idx] = (drow_len[targetRow] + tile_k - 1)/tile_k;
}

__global__ void GetrowPtrbyWarpProduct(int *d_blcPtr, int *rowPtrbyWarp, int blc_row, int cap)
{
    int rowid = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowid >= blc_row)
        return;
    rowPtrbyWarp[rowid] = (d_blcPtr[rowid + 1] - d_blcPtr[rowid] + cap - 1) / cap;
}

__global__ void GetrowIdxbyWarpProduct(int *rowPtrbyWarp, int *rowIdxbyWarp, int blc_row, int tile_m)
{
    int rowid = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowid >= blc_row)
        return;

    int offset = rowPtrbyWarp[rowid];
    int stride = rowPtrbyWarp[rowid + 1] - rowPtrbyWarp[rowid];

    for (int i = offset; i < (offset + stride); i++)
    {
        rowIdxbyWarp[i] = rowid*tile_m;
    }
}

__global__ void GetWarpBlcPtrProduct(int *warpPtr,int *blcPtr, int* warpBlcPtr,int brow, int cap) {
    int rowid = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowid >= brow) return;
    int start = blcPtr[rowid];
    int end = blcPtr[rowid + 1];
    int warpCount = (end - start + cap - 1) / cap;

    for (int j = 1; j <= warpCount; j++) {
        warpBlcPtr[warpPtr[rowid] + j] = min(end, start + j * cap);
    }
}
template<typename IDX_TYPE>
__global__ void csr2AtBsrProductKernel(double* d_val,int *rowInx, int* d_colIdx,int* drow_ptr,int brow,int nrows,int tile_m, int tile_k, int* blc_ptr,double* blc_val, IDX_TYPE *blc_idx) {
    int rowid = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowid >= nrows) return;
    int originrowid = rowid;
    rowid = rowInx[rowid];
    int bid = originrowid/tile_m;
    int pre = blc_ptr[bid];
    for (int i = drow_ptr[rowid], j = 0; i < drow_ptr[rowid+1]; i++,j++) {
        int innerbid = j/tile_k;
        int inneridx = (originrowid%tile_m)*tile_k + j%tile_k;
        blc_val[(pre + innerbid)*32 + inneridx] = d_val[i];
        blc_idx[(pre + innerbid)*32 + inneridx] = IDX_TYPE(d_colIdx[i]);
    }
}

template<typename IDX_TYPE>
__global__ void  csr2AtBsrBalKernelProduct(double* d_val,int* drow_index,int* d_colIdx,int* drow_ptr, int nrows, int warpCount,int* rowIdxbyWarp, int*blcPtr, int* warpBlcPtr,int tile_m,int tile_k,double* blc_val,IDX_TYPE* blc_idx) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = idx >> 5;
    int laneId = threadIdx.x % 32;
    if (warpId >= warpCount) return;
    int targetRow = rowIdxbyWarp[warpId] + laneId/tile_k;
    if (targetRow >= nrows) return;
    int csrrow = drow_index[targetRow];
    int csrStartIdx = drow_ptr[csrrow];
    int csrEndIdx = drow_ptr[csrrow+1];
    int j = (warpBlcPtr[warpId] - blcPtr[targetRow/tile_m])*tile_k + laneId%tile_k;
    for (int i = warpBlcPtr[warpId]; i < warpBlcPtr[warpId+1]; i++,j += tile_k) {
        int csridx = csrStartIdx + j;
        int bsridx = i*32 + laneId;
        blc_val[bsridx] = csridx < csrEndIdx ? d_val[csridx]: 0;
        blc_idx[bsridx] = csridx < csrEndIdx ? IDX_TYPE(d_colIdx[csridx]) :-1;
    }
}

void prepareAtBSRProductRowOrder(const CSRMatrix<double, int32_t>& csr, int* lengths, int* indices) {
    if (csr.nrows < 0 || (csr.nrows && (!csr.row_ptr || !lengths || !indices)))
        throw std::invalid_argument("invalid row order input");
    if (!csr.nrows) return;
    thrust::sequence(thrust::device, indices, indices + csr.nrows, 0);
    thrust::transform(thrust::device, indices, indices + csr.nrows, lengths, RowNNZCalculator(csr.row_ptr));
    thrust::sort_by_key(thrust::device, lengths, lengths + csr.nrows, indices);
}

static int packAtBSRProduct(CSRMatrix<double, int32_t>&csrMat, AtBsr<double, int32_t>&atbsr, int idx, int balance,
                            const int* sorted_lengths, const int* sorted_indices, AtBSRProductStatsCache* stats_cache) {
    atbsr.kernelConfig = atspmv::kernelConfigFromEnvironment();
    const auto config = atbsr.kernelConfig;
    if (idx < -1 || idx > 5 || balance < -1 || balance > 1)
        throw std::invalid_argument("invalid forced tile/mode");
    if (csrMat.nnz == 0 || csrMat.nrows == 0) {
        atbsr.nrows = csrMat.nrows;
        atbsr.ncols = csrMat.ncols;
        atbsr.nnz = 0;
        atbsr.useBalance = balance == 1;
        atbsr.TILE_M = 1 << (idx < 0 ? 0 : idx);
        atbsr.TILE_K = 32 / atbsr.TILE_M;
        atbsr.warpCount = atbsr.blockCount = 0;
        atbsr.predictedTimeUs = 0;
        return 0;
    }
    // check use int16
    double *d_val = csrMat.val;
    int *d_colIdx = csrMat.col_idx;
    int *d_rowPtr = csrMat.row_ptr;
#ifdef DEBUG
#endif

    // TODO
    csrMat.useColInt16 = csrMat.ncols < INT16_MAX;
    atbsr.useColIdxInt16 = csrMat.useColInt16;
    if ((sorted_lengths == nullptr) != (sorted_indices == nullptr))
        throw std::invalid_argument("row order requires both lengths and indices");
    // Construction can fail partway through (e.g. an out-of-memory error or
    // an unsupported tile). Keep temporary and not-yet-transferred buffers
    // owned until the packed descriptor has been fully initialized.
    struct PendingAllocations {
        std::vector<void*> pointers;
        PendingAllocations() { pointers.reserve(8); }
        ~PendingAllocations() { for(void* p:pointers) if(p) cudaFree(p); }
        void own(void* p) { pointers.push_back(p); }
        void release(void* p) { for(auto& owned:pointers) if(owned==p) {owned=nullptr;return;} }
    } pending;
    auto alloc=[&](auto** p,size_t bytes) {
        atspmv::modelCudaCheck(cudaMalloc(p,bytes));
        pending.own(*p);
    };
    int *owned_lengths=nullptr, *drow_index;
    const int* drow_len=sorted_lengths;
    alloc(&drow_index, csrMat.nrows * sizeof(int));
    if (sorted_lengths) {
        atspmv::modelCudaCheck(cudaMemcpy(drow_index,sorted_indices,csrMat.nrows*sizeof(int),cudaMemcpyDeviceToDevice));
    } else {
        alloc(&owned_lengths, csrMat.nrows * sizeof(int));
        prepareAtBSRProductRowOrder(csrMat,owned_lengths,drow_index);
        drow_len=owned_lengths;
    }
#ifdef DEBUG
    printf("sort done\n");
#endif
    const bool cache_hit=stats_cache&&stats_cache->initialized&&stats_cache->tiles_per_task==config.tiles_per_task&&
                         stats_cache->sorted_lengths==sorted_lengths&&stats_cache->rows==csrMat.nrows;
    const auto selection = atspmv::selectTileByTime(
        drow_len, csrMat.nrows, atbsr.useColIdxInt16 ? 2 : 4,
        [&](int i, bool b, bool small) { return configuredResidentWarps(i, b, small, config); },
        idx, balance, csrMat.ncols, csrMat.nnz, config,cache_hit?stats_cache->stats:nullptr);
    if(stats_cache&&!cache_hit) {
        std::copy(selection.stats,selection.stats+6,stats_cache->stats);
        stats_cache->sorted_lengths=sorted_lengths;stats_cache->rows=csrMat.nrows;
        stats_cache->tiles_per_task=config.tiles_per_task;stats_cache->initialized=true;
    }
    const int selected = selection.best.tile_index;
    const int tile_m = 1 << selected;
    const int tile_k = 32 / tile_m;
    atbsr.useBalance = selection.best.balanced;
    atbsr.predictedTimeUs = selection.best.total_ns / 1000.0;
    atbsr.modelResidentWarps = configuredResidentWarps(selected, atbsr.useBalance, atbsr.useColIdxInt16, config);
    atbsr.modelCalibrated = selection.calibrated;
    const int warpCount = static_cast<int>(selection.best.warps);
    const int blockCount = static_cast<int>(selection.stats[selected].tiles);
    // convert csr to atbsr
    int *blc_ptr;
    int brow = (csrMat.nrows + tile_m - 1)/tile_m;
    alloc(&blc_ptr, (brow+1)*sizeof(int));
    atspmv::modelCudaCheck(cudaMemset(blc_ptr, 0, (brow+1)*sizeof(int)));
    double *blc_val;
    int *blc_idx = nullptr;
    int16_t* blc_idxint16 = nullptr;
#ifdef DEBUG
    printf("nonzerorows:%d blockCount:%d warpCount:%d\n",csrMat.nonzeroRows, blockCount, warpCount);
#endif
    alloc(&blc_val, (blockCount*32*sizeof(double)));
    atspmv::modelCudaCheck(cudaMemset(blc_val, 0, (blockCount*32*sizeof(double))));
    // Exactly one index width is used by packing and execution. Keeping both
    // arrays wastes substantial device memory, especially during Auto tuning.
    if (atbsr.useColIdxInt16) {
        alloc(&blc_idxint16, (blockCount*32*sizeof(int16_t)));
        atspmv::modelCudaCheck(cudaMemset(blc_idxint16, 0, (blockCount*32*sizeof(int16_t))));
    } else {
        alloc(&blc_idx, (blockCount*32*sizeof(int)));
        atspmv::modelCudaCheck(cudaMemset(blc_idx, 0, (blockCount*32*sizeof(int))));
    }

    getBlcPtrKernelProduct<<<(brow + 255)/256 ,256>>>(drow_len, tile_m, tile_k, csrMat.nrows, blc_ptr);
    atspmv::modelCudaCheck(cudaDeviceSynchronize());
    thrust::exclusive_scan(thrust::device, blc_ptr, blc_ptr + (brow + 1), blc_ptr, 0);
    int *warpBlcPtr;
    int *rowPtrbyWarp;
    int *rowIdxbyWarp;

    alloc(&rowPtrbyWarp, (brow+1)*sizeof(int));
    atspmv::modelCudaCheck(cudaMemset(rowPtrbyWarp, 0, (brow+1)*sizeof(int)));
    alloc(&rowIdxbyWarp, warpCount*sizeof(int));
    alloc(&warpBlcPtr, (warpCount+1)*sizeof(int));
    atspmv::modelCudaCheck(cudaMemset(warpBlcPtr, 0,  (warpCount+1)*sizeof(int)));
    atspmv::modelCudaCheck(cudaGetLastError());
#ifdef DEBUG
    printf("getBlcPtrKernel done\n");
#endif
    if (atbsr.useBalance) {
        GetrowPtrbyWarpProduct<<<(brow + 255)/256 ,256>>>(blc_ptr, rowPtrbyWarp, brow, config.tiles_per_task);
        atspmv::modelCudaCheck(cudaDeviceSynchronize());
        thrust::exclusive_scan(thrust::device, rowPtrbyWarp, rowPtrbyWarp + brow + 1, rowPtrbyWarp, 0);
        GetrowIdxbyWarpProduct<<<(brow + 255)/256 ,256>>>(rowPtrbyWarp, rowIdxbyWarp, brow, tile_m);
        GetWarpBlcPtrProduct<<<(brow + 255)/256 ,256>>>(rowPtrbyWarp, blc_ptr, warpBlcPtr, brow, config.tiles_per_task);

        atspmv::modelCudaCheck(cudaDeviceSynchronize());
#ifdef DEBUG
    printf("GetWarpBlcPtr done\n");
#endif
        if (atbsr.useColIdxInt16)
            csr2AtBsrBalKernelProduct<int16_t><<<(warpCount + 7)/8 ,256>>>(d_val, drow_index, d_colIdx,d_rowPtr, csrMat.nrows, warpCount, rowIdxbyWarp, blc_ptr, warpBlcPtr, tile_m, tile_k, blc_val, blc_idxint16);
        else
            csr2AtBsrBalKernelProduct<int32_t><<<(warpCount + 7)/8 ,256>>>(d_val, drow_index, d_colIdx, d_rowPtr, csrMat.nrows, warpCount, rowIdxbyWarp, blc_ptr, warpBlcPtr, tile_m, tile_k, blc_val, blc_idx);
    } else {
        if (atbsr.useColIdxInt16)
            csr2AtBsrProductKernel<int16_t><<<(csrMat.nrows + 255)/256 ,256>>>(d_val, drow_index, d_colIdx,  d_rowPtr, brow, csrMat.nrows,tile_m,tile_k, blc_ptr,blc_val,blc_idxint16);
        else
            csr2AtBsrProductKernel<int32_t><<<(csrMat.nrows + 255)/256 ,256>>>(d_val, drow_index, d_colIdx,  d_rowPtr, brow, csrMat.nrows,tile_m,tile_k, blc_ptr,blc_val,blc_idx);
    }
    atspmv::modelCudaCheck(cudaDeviceSynchronize());

#ifdef DEBUG
    printf("csr2AtBsrKernel done\n");
#endif
    atspmv::modelCudaCheck(cudaGetLastError());
#ifdef DEBUG_PTR
    int *h_blcPtr = (int*)malloc(sizeof(int)*(brow+1));
    cudaMemcpy(h_blcPtr, blc_ptr, sizeof(int)*(brow+1), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) {
        printf("blcPtr[%d]:%d\n", i, h_blcPtr[i]);
    }
#endif
    atspmv::modelCudaCheck(cudaGetLastError());

    atbsr.brow = brow;
    atbsr.nrows = csrMat.nrows;
    atbsr.ncols = csrMat.ncols;
    atbsr.nnz = csrMat.nnz;
    atbsr.TILE_M = tile_m;
    atbsr.TILE_K = tile_k;
    atbsr.blockCount = blockCount;
    atbsr.recorder = drow_index;
    atbsr.blcIdx = blc_idx;
    atbsr.blcIdxInt16 = blc_idxint16;
    atbsr.blcVal = blc_val;
    atbsr.blcPtr = blc_ptr;
    atbsr.warpCount = warpCount;
    atbsr.targetRowByWarp = rowIdxbyWarp;
    atbsr.warpBlcPtr = warpBlcPtr;
    // atbsr.recorderCPU = (int*)malloc(sizeof(int)*atbsr.nrows);

    for(void* p:{static_cast<void*>(drow_index),static_cast<void*>(blc_ptr),
                 static_cast<void*>(blc_val),static_cast<void*>(blc_idx),
                 static_cast<void*>(blc_idxint16),static_cast<void*>(rowIdxbyWarp),
                 static_cast<void*>(warpBlcPtr)}) if(p) pending.release(p);

    return 0;
}

int CSRToAtBSRProduct(CSRMatrix<double, int32_t>& csr, AtBsr<double, int32_t>& atbsr, int idx, int balance) {
    return packAtBSRProduct(csr,atbsr,idx,balance,nullptr,nullptr,nullptr);
}

int CSRToAtBSRProductFromRows(CSRMatrix<double, int32_t>& csr, AtBsr<double, int32_t>& atbsr,
                             int idx, int balance, const int* lengths, const int* indices, AtBSRProductStatsCache* stats_cache) {
    if (csr.nnz && (!lengths || !indices)) throw std::invalid_argument("missing precomputed row order");
    return packAtBSRProduct(csr,atbsr,idx,balance,lengths,indices,stats_cache);
}
#define WARP_CAPACITY 32

template<typename IDX_TYPE>
__global__ void TcSPMVKernel1x32CC(int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y,int *recorder) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= warpCount) return;
    int laneId = tid & 31;
    double fragA, fragB, fragC = 0.0;
    for (int i = BlcPtr[warpId]; i < BlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;

    }
    fragC += __shfl_down_sync(0xffffffff, fragC, 1, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 2, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 4, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 8, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 16, 32);

    // fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    // fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);
    if (laneId  == 0) {
        int rowId = warpId;
        if (rowId < nrows) {
            store_double_to_global(&y[recorder[rowId]], fragC);
        }
    }
}

template<typename IDX_TYPE>
__global__ void TcSPMVKernel2x16CC(int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y,int *recorder) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= warpCount) return;
    int laneId = tid & 31;
    double fragA, fragB, fragC = 0.0;
    for (int i = BlcPtr[warpId]; i < BlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;

    }
    fragC += __shfl_down_sync(0xffffffff, fragC, 1, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 2, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 4, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 8, 32);
    // fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    // fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);
    if (laneId  % 16 == 0) {
        int rowId = warpId*2 + laneId/16;
        if (rowId < nrows) {
            // y[rowId] = fragC;
            store_double_to_global(&y[recorder[rowId]], fragC);
        }
    }
}
template<typename IDX_TYPE>
__global__ void TcSPMVKernel4x8CC(int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y,int *recorder) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= warpCount) return;
    int laneId = tid & 31;
    double fragA, fragB, fragC = 0.0;
    for (int i = BlcPtr[warpId]; i < BlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;

    }
    fragC += __shfl_down_sync(0xffffffff, fragC, 1, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 2, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 4, 32);
    if (laneId  % 8 == 0) {
        int rowId = warpId*4 + laneId/8;
        if (rowId < nrows) {
            store_double_to_global(&y[recorder[rowId]], fragC);
        }
    }
}

template<typename IDX_TYPE>
__global__ void TcSPMVKernel8x4CC(int32_t *blcPtr, IDX_TYPE *blcIdx, double *blcVal, int brows, int nrows,  double *x, double *y,int *recorder) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= brows) return;
    int laneId = tid & 31;
    double fragA, fragB, fragC = 0.0;

    #pragma unroll 4
    for (int i = blcPtr[warpId]; i < blcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;
    }

    fragC += __shfl_down_sync(0xffffffff, fragC, 1, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 2, 32);
    if (laneId%4 == 0) {
        int rowId = warpId*8 + laneId/4;
        if (rowId < nrows) {
            store_double_to_global(&y[recorder[rowId]], fragC);

        }
    }

}
template<typename IDX_TYPE>
__global__ void TcSPMVKernel16x2CC(int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y,int *recorder) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= warpCount) return;
    int laneId = tid & 31;
    double fragA, fragB, fragC = 0.0;
    for (int i = BlcPtr[warpId]; i < BlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;

    }

    fragC += __shfl_down_sync(0xffffffff, fragC, 1, 32);
    if (laneId  % 2 == 0) {
        int rowId = warpId*16 + laneId/2;
        if (rowId < nrows) {
            // y[rowId] = fragC;
                store_double_to_global(&y[recorder[rowId]], fragC);
        }
    }
}

template<typename IDX_TYPE>
__global__ void TcSPMVKernel32x1CC(int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y, int *recorder) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= warpCount) return;
    int laneId = tid & 31;
    int rowId = warpId*32 + laneId;

    double fragA, fragB, fragC = 0.0;

    for (int i = BlcPtr[warpId]; i < BlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;

    }

    if (rowId < nrows) {
        // y[rowId] = fragC;
        store_double_to_global(&y[recorder[rowId]], fragC);
    }
}
template<typename IDX_TYPE>
__global__ void TcSPMVKernel32x1Bal(int32_t* warpTargetRow, int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y,int *recorder) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= warpCount) return;
    int laneId = tid & 31;
    int warpRowId = warpTargetRow[warpId];
    int rowId = warpRowId + laneId;

    double fragA, fragB, fragC = 0.0;

    for (int i = BlcPtr[warpId]; i < BlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;
    }

    if (rowId < nrows) {
        atomicAdd(&y[recorder[rowId]], fragC);
    }
}
template<typename IDX_TYPE>
__global__ void TcSPMVKernel16x2Bal(int32_t* warpTargetRow, int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y,int *recorder) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= warpCount) return;
    int laneId = tid & 31;
    double fragA, fragB, fragC = 0.0;
    for (int i = BlcPtr[warpId]; i < BlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;

    }
    int warpRowId = warpTargetRow[warpId];
    fragC += __shfl_down_sync(0xffffffff, fragC, 1, 32);
    if (laneId  % 2 == 0) {

        int rowId = warpRowId + laneId/2;

        if (rowId < nrows) {
            // y[rowId] = fragC;
            atomicAdd(&y[recorder[rowId]], fragC);
        }
    }
}
template<typename IDX_TYPE>
__global__ void TcSPMVKernel8x4Bal(int32_t *warpTargetRow, int32_t *warpBlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y,int *recorder) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= warpCount) return;
    int laneId = tid & 31;
    double fragA, fragB, fragC = 0.0;
    // fragC[2] ={0};
    int targetRow = warpTargetRow[warpId];
    #pragma unroll 4
    for (int i = warpBlcPtr[warpId]; i < warpBlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;
    }
    fragC += __shfl_down_sync(0xffffffff, fragC, 1, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 2, 32);
    if (laneId%4 == 0) {
        int rowId = targetRow + laneId/4;
        if (rowId < nrows) {
            atomicAdd(&y[recorder[rowId]], fragC);
            // y[rowId] = fragC;
        }
    }
}
template<typename IDX_TYPE>
__global__ void TcSPMVKernel4x8Bal(int32_t *warpTargetRow, int32_t *warpBlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y,int *recorder) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= warpCount) return;
    int laneId = tid & 31;
    double fragA, fragB, fragC = 0.0;
    for (int i = warpBlcPtr[warpId]; i < warpBlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;
    }
    int warpRowId = warpTargetRow[warpId];
    fragC += __shfl_down_sync(0xffffffff, fragC, 1, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 2, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 4, 32);
    if (laneId % 8 == 0) {
        int rowId = warpRowId + laneId/8;
        if (rowId < nrows) {
            atomicAdd(&y[recorder[rowId]], fragC);
        }
    }
}

template<typename IDX_TYPE>
__global__ void TcSPMVKernel2x16Bal(int32_t* warpTargetRow, int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y, int *recorder) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= warpCount) return;
    int laneId = tid & 31;
    double fragA, fragB, fragC = 0.0;
    for (int i = BlcPtr[warpId]; i < BlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;

    }
    int warpRowId = warpTargetRow[warpId];
    fragC += __shfl_down_sync(0xffffffff, fragC, 1, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 2, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 4, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 8, 32);
    if (laneId  % 16 == 0) {
        int rowId = warpRowId + laneId/16;
        if (rowId < nrows) {
            // y[rowId] = fragC;
            atomicAdd(&y[recorder[rowId]], fragC);
        }
    }
}

template<typename IDX_TYPE>
__global__ void TcSPMVKernel1x32Bal(int32_t* warpTargetRow, int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y, int* recorder) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= warpCount) return;
    int laneId = tid & 31;
    double fragA, fragB, fragC = 0.0;

    int warpRowId = warpTargetRow[warpId];

    for (int i = BlcPtr[warpId]; i < BlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        // fragA =1.0;
        // fragB = 1.0;
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;
    }

    fragC += __shfl_down_sync(0xffffffff, fragC, 1, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 2, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 4, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 8, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 16, 32);

    if (laneId  == 0) {
        int rowId = warpRowId;
        if (rowId < nrows) {
            // y[rowId] = fragC;
            atomicAdd(&y[recorder[warpRowId]], fragC);
        }
    }
}

// One mapping serves occupancy prediction and legacy execution.
template<class Index>
static const void* legacyProductKernel(int index, bool balanced) {
    switch (index) {
    case 0:
        if (balanced) return reinterpret_cast<const void*>(TcSPMVKernel1x32Bal<Index>);
        return reinterpret_cast<const void*>(TcSPMVKernel1x32CC<Index>);
    case 1:
        if (balanced) return reinterpret_cast<const void*>(TcSPMVKernel2x16Bal<Index>);
        return reinterpret_cast<const void*>(TcSPMVKernel2x16CC<Index>);
    case 2:
        if (balanced) return reinterpret_cast<const void*>(TcSPMVKernel4x8Bal<Index>);
        return reinterpret_cast<const void*>(TcSPMVKernel4x8CC<Index>);
    case 3:
        if (balanced) return reinterpret_cast<const void*>(TcSPMVKernel8x4Bal<Index>);
        return reinterpret_cast<const void*>(TcSPMVKernel8x4CC<Index>);
    case 4:
        if (balanced) return reinterpret_cast<const void*>(TcSPMVKernel16x2Bal<Index>);
        return reinterpret_cast<const void*>(TcSPMVKernel16x2CC<Index>);
    case 5:
        if (balanced) return reinterpret_cast<const void*>(TcSPMVKernel32x1Bal<Index>);
        return reinterpret_cast<const void*>(TcSPMVKernel32x1CC<Index>);
    default: throw std::invalid_argument("invalid tile index");
    }
}
static bool usesTunedKernel(bool balanced, const atspmv::KernelConfig& config) {
    return config.prefetch != 0 || (config.cta_reduce && balanced);
}
static const void* productKernel(int index, bool balanced, bool small,
                                 const atspmv::KernelConfig& config) {
    if (usesTunedKernel(balanced, config))
        return atspmv::tunedKernel(index, balanced, small, config);
    return small ? legacyProductKernel<int16_t>(index, balanced)
                 : legacyProductKernel<int32_t>(index, balanced);
}
static int configuredResidentWarps(int index, bool balanced, bool small,
                                   const atspmv::KernelConfig& config) {
    int blocks = 0;
    atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, productKernel(index, balanced, small, config),
        config.warps_per_block * 32, 0));
    return blocks * config.warps_per_block;
}

__global__ void beta_vecY_prod(double *d_y, double beta, int row)
{
    int rowid = threadIdx.x + blockDim.x * blockIdx.x;
    if (rowid >= row)
        return;
    d_y[rowid] = beta == 0.0 ? 0.0 : d_y[rowid] * beta;
}
int AtSpmvProduct(AtBsr<double, int32_t>& atbsr, double* d_x, double* d_y, double alpha, double beta, cudaStream_t stream) {

    if (atbsr.nrows == 0) return 0;
    if (atbsr.nnz == 0) {
        if (beta != 1.0)
            beta_vecY_prod<<<(atbsr.nrows+127)/128, 128, 0, stream>>>(d_y, beta, atbsr.nrows);
        return 0;
    }
    int tile_m = atbsr.TILE_M;
#ifdef DEBUG
    printf("AtSpmv\n");
    printf("tile_m %d tile_k %d bal: %d int16:%d\n", tile_m, atbsr.TILE_K, atbsr.useBalance, atbsr.useColIdxInt16);
#endif
    int blockSize = atbsr.kernelConfig.warps_per_block * WARP_SIZE;
    int gridSize = (atbsr.warpCount + atbsr.kernelConfig.warps_per_block - 1) / atbsr.kernelConfig.warps_per_block;
    // Normal mode overwrites every row, including empty rows. For beta=0
    // a separate clearing launch is redundant; split tasks still need it.
    if (beta != 1.0 && (atbsr.useBalance || beta != 0.0)) {
        beta_vecY_prod<<<(atbsr.nrows+127)/128, 128, 0, stream>>>(d_y, beta, atbsr.nrows);
    }
    int index = 0;
    while ((1 << index) < tile_m) ++index;
    const void* kernel = productKernel(index, atbsr.useBalance,
                                       atbsr.useColIdxInt16, atbsr.kernelConfig);
    int* ptr = atbsr.useBalance ? atbsr.warpBlcPtr : atbsr.blcPtr;
    void* columns = atbsr.useColIdxInt16 ? static_cast<void*>(atbsr.blcIdxInt16)
                                        : static_cast<void*>(atbsr.blcIdx);
    if (atbsr.useBalance || usesTunedKernel(atbsr.useBalance, atbsr.kernelConfig)) {
        void* args[] = {&atbsr.targetRowByWarp, &ptr, &columns, &atbsr.blcVal,
                        &atbsr.warpCount, &atbsr.nrows, &d_x, &d_y, &atbsr.recorder};
        atspmv::modelCudaCheck(cudaLaunchKernel(kernel, dim3(gridSize), dim3(blockSize), args, 0, stream));
    } else {
        void* args[] = {&ptr, &columns, &atbsr.blcVal, &atbsr.brow,
                        &atbsr.nrows, &d_x, &d_y, &atbsr.recorder};
        atspmv::modelCudaCheck(cudaLaunchKernel(kernel, dim3(gridSize), dim3(blockSize), args, 0, stream));
    }
    return 0;
}
