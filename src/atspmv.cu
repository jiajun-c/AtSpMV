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
#include "matrixFormat.h"
#include "common.h"
#include "ptx.h"
#include "tileSelector.cuh"
#include "atspmvprod.h"

static int tileResidentWarps(int index, bool balanced, bool int16);

using atspmv::detail::RowNNZCalculator;

int getMaxPhyThread(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    size_t total_concurrent_threads = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
    return total_concurrent_threads;
}
__global__ void getBlcPtrKernel(int *drow_len,int tile_m,int tile_k, int nrows, int* blc_ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (nrows + tile_m - 1)/tile_m) return;
    int targetRow = min(nrows-1, (idx+1)*tile_m - 1);
    blc_ptr[idx] = (drow_len[targetRow] + tile_k - 1)/tile_k;
}

__global__ void GetrowPtrbyWarp(int *d_blcPtr, int *rowPtrbyWarp, int blc_row)
{
    int rowid = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowid >= blc_row)
        return;
    rowPtrbyWarp[rowid] = (d_blcPtr[rowid + 1] - d_blcPtr[rowid] + 8 - 1) / 8;
}

__global__ void GetrowIdxbyWarp(int *rowPtrbyWarp, int *rowIdxbyWarp, int blc_row, int tile_m)
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

__global__ void GetWarpBlcPtr(int *warpPtr,int *blcPtr, int* warpBlcPtr,int brow) {
    int rowid = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowid >= brow) return;
    int start = blcPtr[rowid];
    int end = blcPtr[rowid + 1];
    int warpCount = (end - start + 8 - 1) / 8;

    for (int j = 1; j <= warpCount; j++) {
        warpBlcPtr[warpPtr[rowid] + j] = min(end, start + j * 8);
    }
}
template<typename IDX_TYPE>
__global__ void csr2AtBsrKernel(double* d_val,int *rowInx, int* d_colIdx,int* drow_ptr,int brow,int nrows,int tile_m, int tile_k, int* blc_ptr,double* blc_val, IDX_TYPE *blc_idx) {
    int rowid = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowid >= nrows) return;
    int originrowid = rowid;
    rowid = rowInx[rowid];
    int bid = originrowid/tile_m;
    int pre = blc_ptr[bid];
    for (int i = drow_ptr[rowid], j = 0; i < drow_ptr[rowid+1]; i++,j++) {
        int innerbid = j/tile_k;
        int inneridx = (originrowid%tile_m)*tile_k + j%tile_k;
        blc_val[(pre + innerbid )*32 + inneridx] = d_val[i];
        blc_idx[(pre + innerbid)*32 + inneridx] = IDX_TYPE(d_colIdx[i]);
    }
}

template<typename IDX_TYPE>
__global__ void  csr2AtBsrBalKernel(double* d_val,int* drow_index,int* d_colIdx,int* drow_ptr, int nrows, int warpCount,int* rowIdxbyWarp, int*blcPtr, int* warpBlcPtr,int tile_m,int tile_k,double* blc_val,IDX_TYPE* blc_idx) {
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

__host__ int CSRToAtBSR(CSRMatrix<double, int32_t>&csrMat, AtBsr<double, int32_t>&atbsr, double& preTime, int idx, int64_t &buffsize) {
    // check use int16
    double *d_val;
    int *d_colIdx;
    int *d_rowPtr;
#ifdef DEBUG
#endif

    CHECK(cudaMalloc((void**)&d_val, csrMat.nnz * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_colIdx, csrMat.nnz * sizeof(int)));
    CHECK(cudaMalloc((void**)&d_rowPtr, (csrMat.nrows + 1) * sizeof(int)));
    CHECK(cudaMemcpy(d_val, csrMat.val, csrMat.nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_colIdx, csrMat.col_idx, csrMat.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_rowPtr, csrMat.row_ptr, (csrMat.nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    auto start = std::chrono::high_resolution_clock::now();

    // TODO
    csrMat.useColInt16 = csrMat.ncols < INT16_MAX;
    atbsr.useColIdxInt16 = csrMat.useColInt16;
    int *drow_len, *drow_index;
    CHECK(cudaMalloc((void**)&drow_len, csrMat.nrows * sizeof(int)));
    CHECK(cudaMalloc((void**)&drow_index, csrMat.nrows * sizeof(int)));
    thrust::sequence(thrust::device, drow_index, drow_index + csrMat.nrows, 0);
    thrust::transform(thrust::device,drow_index, drow_index + csrMat.nrows, drow_len, RowNNZCalculator(d_rowPtr));
    thrust::sort_by_key(thrust::device,
        drow_len, drow_len + csrMat.nrows,  // 排序的键值
        drow_index                    // 对应的值随之重排
    );
#ifdef DEBUG
    printf("sort done\n");
#endif
    const auto selection = atspmv::selectTileByTime(
        drow_len, csrMat.nrows, atbsr.useColIdxInt16 ? 2 : 4,
        tileResidentWarps, idx, -1, csrMat.ncols, csrMat.nnz);
    const int selected = selection.best.tile_index;
    const int tile_m = 1 << selected;
    const int tile_k = 32 / tile_m;
    atbsr.useBalance = selection.best.balanced;
    atbsr.predictedTimeUs = selection.best.total_ns / 1000.0;
    atbsr.modelCalibrated = selection.calibrated;
    atbsr.modelResidentWarps = tileResidentWarps(selected, atbsr.useBalance, atbsr.useColIdxInt16);
    const int warpCount = static_cast<int>(selection.best.warps);
    const int blockCount = static_cast<int>(selection.stats[selected].tiles);
    // convert csr to atbsr
    int *blc_ptr;
    int brow = (csrMat.nrows + tile_m - 1)/tile_m;
    cudaMalloc(&blc_ptr, (brow+1)*sizeof(int));
    cudaMemset(blc_ptr, 0, (brow+1)*sizeof(int));
    double *blc_val;
    int *blc_idx;
    int16_t* blc_idxint16;
#ifdef DEBUG
    printf("nonzerorows:%d blockCount:%d warpCount:%d\n",csrMat.nonzeroRows, blockCount, warpCount);
#endif
    cudaMalloc(&blc_val, (blockCount*32*sizeof(double)));
    cudaMemset(blc_val, 0, (blockCount*32*sizeof(double)));
    cudaMalloc(&blc_idx, (blockCount*32*sizeof(int)));
    cudaMalloc(&blc_idxint16, (blockCount*32*sizeof(int16_t)));

    cudaMemset(blc_idx, 0, (blockCount*32*sizeof(int)));
    cudaMemset(blc_idxint16, 0, (blockCount*32*sizeof(int16_t)));
    if (csrMat.useColInt16) {
        buffsize = (blockCount*32*sizeof(int16_t)) + (blockCount*32*sizeof(double)) + (brow+1+csrMat.nrows)*sizeof(int);
    } else {
        buffsize = (blockCount*32*sizeof(int)) + (blockCount*32*sizeof(double)) + (brow+1 + csrMat.nrows)*sizeof(int);
    }
    if (atbsr.useBalance) {
        buffsize += (brow+1 + warpCount*2+1)*sizeof(int);
    }
    getBlcPtrKernel<<<(brow + 255)/256 ,256>>>(drow_len, tile_m, tile_k, csrMat.nrows, blc_ptr);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, blc_ptr, blc_ptr + (brow + 1), blc_ptr, 0);
    int *warpBlcPtr;
    int *rowPtrbyWarp;
    int *rowIdxbyWarp;

    cudaMalloc(&rowPtrbyWarp, (brow+1)*sizeof(int));
    cudaMemset(rowPtrbyWarp, 0, (brow+1)*sizeof(int));
    cudaMalloc(&rowIdxbyWarp, warpCount*sizeof(int));
    cudaMalloc(&warpBlcPtr, (warpCount+1)*sizeof(int));
    cudaMemset(warpBlcPtr, 0,  (warpCount+1)*sizeof(int));
    CHECK(cudaGetLastError());
#ifdef DEBUG
    printf("getBlcPtrKernel done\n");
#endif
    if (atbsr.useBalance) {
        GetrowPtrbyWarp<<<(brow + 255)/256 ,256>>>(blc_ptr, rowPtrbyWarp, brow);
        cudaDeviceSynchronize();
        thrust::exclusive_scan(thrust::device, rowPtrbyWarp, rowPtrbyWarp + brow + 1, rowPtrbyWarp, 0);
        GetrowIdxbyWarp<<<(brow + 255)/256 ,256>>>(rowPtrbyWarp, rowIdxbyWarp, brow, tile_m);
        GetWarpBlcPtr<<<(brow + 255)/256 ,256>>>(rowPtrbyWarp, blc_ptr, warpBlcPtr, brow);
        cudaDeviceSynchronize();

#ifdef DEBUG
    printf("GetWarpBlcPtr done\n");
#endif
        if (atbsr.useColIdxInt16)
            csr2AtBsrBalKernel<int16_t><<<(warpCount + 7)/8 ,256>>>(d_val, drow_index, d_colIdx,d_rowPtr, csrMat.nrows, warpCount, rowIdxbyWarp, blc_ptr, warpBlcPtr, tile_m, tile_k, blc_val, blc_idxint16);
        else
            csr2AtBsrBalKernel<int32_t><<<(warpCount + 7)/8 ,256>>>(d_val, drow_index, d_colIdx, d_rowPtr, csrMat.nrows, warpCount, rowIdxbyWarp, blc_ptr, warpBlcPtr, tile_m, tile_k, blc_val, blc_idx);
    } else {
        if (atbsr.useColIdxInt16)
            csr2AtBsrKernel<int16_t><<<(csrMat.nrows + 255)/256 ,256>>>(d_val, drow_index, d_colIdx,  d_rowPtr, brow, csrMat.nrows,tile_m,tile_k, blc_ptr,blc_val,blc_idxint16);
        else
            csr2AtBsrKernel<int32_t><<<(csrMat.nrows + 255)/256 ,256>>>(d_val, drow_index, d_colIdx,  d_rowPtr, brow, csrMat.nrows,tile_m,tile_k, blc_ptr,blc_val,blc_idx);
    }
    cudaDeviceSynchronize();

#ifdef DEBUG
    printf("csr2AtBsrKernel done\n");
#endif
    CHECK(cudaGetLastError());
#ifdef DEBUG_PTR
    int *h_blcPtr = (int*)malloc(sizeof(int)*(brow+1));
    cudaMemcpy(h_blcPtr, blc_ptr, sizeof(int)*(brow+1), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) {
        printf("blcPtr[%d]:%d\n", i, h_blcPtr[i]);
    }
#endif
    CHECK(cudaGetLastError());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    preTime = duration.count();

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
    atbsr.recorderCPU = (int*)malloc(sizeof(int)*atbsr.nrows);

    cudaMemcpy(atbsr.recorderCPU, drow_index, sizeof(int)*atbsr.nrows, cudaMemcpyDeviceToHost);

    cudaFree(drow_len);
    cudaFree(rowPtrbyWarp);
    cudaFree(d_val);
    cudaFree(d_colIdx);
    cudaFree(d_rowPtr);

    return 0;
}
#define WARP_CAPACITY 32

template <typename IDX_TYPE>
__global__ void spmv_fp64(int32_t *rowPtr, IDX_TYPE *colIdx, int32_t rows, int32_t nnz, double* values,double* x, double* outvalues) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= rows) return;
    double tmp = 0.0;
    #pragma unroll
    for (int32_t i = rowPtr[tid]; i < rowPtr[tid+1]; i++) {
        int32_t col = colIdx[i];
        double val = values[i];
        tmp += val *x[col];
    }
    outvalues[tid] = tmp;
}

template <typename IDX_TYPE>
__global__ void spmv_fp64_fixLen(int32_t fixLen, IDX_TYPE *colIdx, int32_t rows, int32_t nnz, double* values,double* x, double* outvalues) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= rows) return;
    double tmp = 0.0;
    #pragma unroll
    for (int32_t i =tid*fixLen; i < tid*fixLen + fixLen; i++) {
        int32_t col = colIdx[i];
        double val = values[i];
        tmp += val *x[col];
    }
    outvalues[tid] = tmp;
}
int spmv_fp64_warpper(CSRMatrix<double, int32_t>&csr ,double* x, double* outvalues, int warmup, int repeat, double& avg_time) {
#ifdef DEBUG
    printf("spmv_fp64_warpper\n");
#endif
    int nnz = csr.nnz;
    int cols = csr.ncols;
    int rows = csr.nrows;
    double *d_values, *d_x, *d_outvalues;
    cudaMalloc((void**)&d_values, sizeof(double)*nnz);
    cudaMalloc((void**)&d_x, sizeof(double)*cols);
    cudaMalloc((void**)&d_outvalues, sizeof(double)*rows);
    cudaMemcpy(d_values, csr.val, sizeof(double)*nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, sizeof(double)*cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_outvalues, outvalues, sizeof(double)*rows, cudaMemcpyHostToDevice);

    int32_t *d_rowPtr, *d_colIdx;
    cudaMalloc((void**)&d_rowPtr, sizeof(int32_t)*(rows+1));
    cudaMalloc((void**)&d_colIdx, sizeof(int32_t)*nnz);
    cudaMemcpy(d_rowPtr, csr.row_ptr, sizeof(int32_t)*(rows+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, csr.col_idx, sizeof(int32_t)*nnz, cudaMemcpyHostToDevice);
    int thread_num = 64;
    int block_num = (rows + thread_num - 1)/thread_num;
    CHECK(cudaGetLastError());
    if (!csr.useColInt16) {
        if (csr.useFixLen) {

            for (int i = 0; i < warmup; i++) {
                spmv_fp64_fixLen<int32_t><<<block_num, thread_num>>>(csr.FixLen, d_colIdx, rows, nnz, d_values, d_x, d_outvalues);
                cudaDeviceSynchronize();
            }
            auto start = chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; i++) {
                spmv_fp64_fixLen<int32_t><<<block_num, thread_num>>>(csr.FixLen, d_colIdx, rows, nnz, d_values, d_x, d_outvalues);
                cudaDeviceSynchronize();
            }
            auto end = chrono::high_resolution_clock::now();
            auto eplapsed = end - start;
            avg_time = chrono::duration_cast<chrono::microseconds>(eplapsed).count() / repeat;
        } else {
            for (int i = 0; i < warmup; i++) {
                spmv_fp64<int32_t><<<block_num, thread_num>>>(d_rowPtr, d_colIdx, rows, nnz, d_values, d_x, d_outvalues);
                cudaDeviceSynchronize();
            }
            auto start = chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; i++) {
                spmv_fp64<int32_t><<<block_num, thread_num>>>(d_rowPtr, d_colIdx, rows, nnz, d_values, d_x, d_outvalues);
                cudaDeviceSynchronize();
            }
            auto end = chrono::high_resolution_clock::now();
            auto eplapsed = end - start;
            avg_time = chrono::duration_cast<chrono::microseconds>(eplapsed).count() / repeat;
        }
    } else {
        int16_t *colIdx16 = (int16_t*)malloc(sizeof(int16_t)*nnz);
        for (int i = 0; i < nnz; i++) {
            colIdx16[i] = csr.col_idx[i];
        }
        cudaMemcpy(d_colIdx, colIdx16, sizeof(int16_t)*nnz, cudaMemcpyHostToDevice);
        free(colIdx16);
        int16_t* d_colIdx16 = reinterpret_cast<int16_t*>(d_colIdx);
        if (csr.useFixLen) {
            for (int i = 0; i < warmup; i++) {
                spmv_fp64_fixLen<int16_t><<<block_num, thread_num>>>(csr.FixLen, d_colIdx16, rows, nnz, d_values, d_x, d_outvalues);
                cudaDeviceSynchronize();
            }
            auto start = chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; i++) {
                spmv_fp64_fixLen<int16_t><<<block_num, thread_num>>>(csr.FixLen, d_colIdx16, rows, nnz, d_values, d_x, d_outvalues);
                cudaDeviceSynchronize();
            }
            auto end = chrono::high_resolution_clock::now();
            auto eplapsed = end - start;
            avg_time = chrono::duration_cast<chrono::microseconds>(eplapsed).count() / repeat;
        } else {
            for (int i = 0; i < warmup; i++) {
                spmv_fp64<int16_t><<<block_num, thread_num>>>(d_rowPtr, d_colIdx16, rows, nnz, d_values, d_x, d_outvalues);
                cudaDeviceSynchronize();
            }
            auto start = chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; i++) {
                spmv_fp64<int16_t><<<block_num, thread_num>>>(d_rowPtr, d_colIdx16, rows, nnz, d_values, d_x, d_outvalues);
                cudaDeviceSynchronize();
            }
            auto end = chrono::high_resolution_clock::now();
            auto eplapsed = end - start;
            avg_time = chrono::duration_cast<chrono::microseconds>(eplapsed).count() / repeat;
        }
    }

    cudaMemcpy(outvalues, d_outvalues, sizeof(double)*rows, cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_outvalues);
    return 0;
}

template<typename IDX_TYPE>
__global__ void TcSPMVKernel1x32CC(int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y) {
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
            // y[rowId] = fragC;
                store_double_to_global(&y[rowId], fragC);

            // atomicAdd(&y[rowId], fragC[0]);
        }
    }
}

template<typename IDX_TYPE>
__global__ void TcSPMVKernel2x16CC(int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y) {
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
                store_double_to_global(&y[rowId], fragC);
        }
    }
}
template<typename IDX_TYPE>
__global__ void TcSPMVKernel4x8CC(int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y) {
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
    // fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    // fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);
    if (laneId  % 8 == 0) {
        int rowId = warpId*4 + laneId/8;
        if (rowId < nrows) {
            store_double_to_global(&y[rowId], fragC);
            // y[rowId] = fragC;
            // atomicAdd(&y[rowId], fragC[0]);
        }
    }
}

template<typename IDX_TYPE>
__global__ void TcSPMVKernel8x4CC(int32_t *blcPtr, IDX_TYPE *blcIdx, double *blcVal, int brows, int nrows,  double *x, double *y) {
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
            // atomicAdd(&y[rowId], fragC[0]);
            // y[rowId] = fragC;
            store_double_to_global(&y[rowId], fragC);

        }
    }

}
template<typename IDX_TYPE>
__global__ void TcSPMVKernel16x2CC(int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y) {
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
                store_double_to_global(&y[rowId], fragC);
        }
    }
}

template<typename IDX_TYPE>
__global__ void TcSPMVKernel32x1CC(int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y) {
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
        store_double_to_global(&y[rowId], fragC);
    }
}
template<typename IDX_TYPE>
__global__ void TcSPMVKernel32x1Bal(int32_t* warpTargetRow, int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y) {
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
            // y[rowId] = fragC;
            atomicAdd(&y[rowId], fragC);
    }
}
template<typename IDX_TYPE>
__global__ void TcSPMVKernel16x2Bal(int32_t* warpTargetRow, int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y) {
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
            atomicAdd(&y[rowId], fragC);
        }
    }
}
template<typename IDX_TYPE>
__global__ void TcSPMVKernel8x4Bal(int32_t *warpTargetRow, int32_t *warpBlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y) {
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
            atomicAdd(&y[rowId], fragC);
            // y[rowId] = fragC;
        }
    }
}
template<typename IDX_TYPE>
__global__ void TcSPMVKernel4x8Bal(int32_t *warpTargetRow, int32_t *warpBlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y) {
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
            atomicAdd(&y[rowId], fragC);
        }
    }
}

template<typename IDX_TYPE>
__global__ void TcSPMVKernel2x16Bal(int32_t* warpTargetRow, int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y) {
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
    // fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    // fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);
    if (laneId  % 16 == 0) {
        int rowId = warpRowId + laneId/16;
        if (rowId < nrows) {
            // y[rowId] = fragC;
            atomicAdd(&y[rowId], fragC);
        }
    }
}

template<typename IDX_TYPE>
__global__ void TcSPMVKernel1x32Bal(int32_t* warpTargetRow, int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y) {
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
            atomicAdd(&y[warpRowId], fragC);
        }
    }
}

static int tileResidentWarps(int index, bool balanced, bool int16) {
    int blocks = 0;
    switch (index) {
    case 0:
        if (balanced) {
            if (int16)
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel1x32Bal<int16_t>, 128, 0));
            else
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel1x32Bal<int32_t>, 128, 0));
        } else {
            if (int16)
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel1x32CC<int16_t>, 128, 0));
            else
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel1x32CC<int32_t>, 128, 0));
        }
        break;
    case 1:
        if (balanced) {
            if (int16)
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel2x16Bal<int16_t>, 128, 0));
            else
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel2x16Bal<int32_t>, 128, 0));
        } else {
            if (int16)
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel2x16CC<int16_t>, 128, 0));
            else
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel2x16CC<int32_t>, 128, 0));
        }
        break;
    case 2:
        if (balanced) {
            if (int16)
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel4x8Bal<int16_t>, 128, 0));
            else
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel4x8Bal<int32_t>, 128, 0));
        } else {
            if (int16)
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel4x8CC<int16_t>, 128, 0));
            else
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel4x8CC<int32_t>, 128, 0));
        }
        break;
    case 3:
        if (balanced) {
            if (int16)
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel8x4Bal<int16_t>, 128, 0));
            else
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel8x4Bal<int32_t>, 128, 0));
        } else {
            if (int16)
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel8x4CC<int16_t>, 128, 0));
            else
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel8x4CC<int32_t>, 128, 0));
        }
        break;
    case 4:
        if (balanced) {
            if (int16)
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel16x2Bal<int16_t>, 128, 0));
            else
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel16x2Bal<int32_t>, 128, 0));
        } else {
            if (int16)
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel16x2CC<int16_t>, 128, 0));
            else
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel16x2CC<int32_t>, 128, 0));
        }
        break;
    case 5:
        if (balanced) {
            if (int16)
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel32x1Bal<int16_t>, 128, 0));
            else
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel32x1Bal<int32_t>, 128, 0));
        } else {
            if (int16)
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel32x1CC<int16_t>, 128, 0));
            else
                atspmv::modelCudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, TcSPMVKernel32x1CC<int32_t>, 128, 0));
        }
        break;
    default: throw std::invalid_argument("invalid tile index");
    }
    return blocks * atspmv::kWarpsPerBlock;
}

__global__ void beta_vecY(double *d_y, double beta, int row)
{
    int rowid = threadIdx.x + blockDim.x * blockIdx.x;
    if (rowid >= row)
        return;
    d_y[rowid] *= beta;
}
int AtSpmv(AtBsr<double, int32_t>& atbsr, double* x, double* y, int warmup, int repeat, double &avgTime) {

    int tile_m = atbsr.TILE_M;
#ifdef DEBUG
    printf("AtSpmv\n");
    printf("tile_m %d tile_k %d bal: %d int16:%d\n", tile_m, atbsr.TILE_K, atbsr.useBalance, atbsr.useColIdxInt16);
#endif
    double* d_x, *d_y;
    cudaMalloc((void**)&d_x, atbsr.ncols * sizeof(double));
    cudaMalloc((void**)&d_y, atbsr.nrows * sizeof(double));

    cudaMemcpy(d_x, x, atbsr.ncols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, atbsr.nrows * sizeof(double), cudaMemcpyHostToDevice);
    int blockSize = 4*WARP_SIZE;
    int gridSize = (atbsr.warpCount + 3) / 4;

    if (atbsr.useBalance) {
        if (tile_m == 1) {
            if (atbsr.useColIdxInt16) {
                for (int i = 0; i < warmup; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel1x32Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel1x32Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            } else {
                for (int i = 0; i < warmup; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel1x32Bal<int32_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel1x32Bal<int32_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            }
        } else if (tile_m == 2) {
            if (atbsr.useColIdxInt16) {
                for (int i = 0; i < warmup; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel2x16Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel2x16Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            } else {
                for (int i = 0; i < warmup; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel2x16Bal<int32_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel2x16Bal<int32_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            }
        } else if (tile_m == 4) {
           if (atbsr.useColIdxInt16) {
                for (int i = 0; i < warmup; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel4x8Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel4x8Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            } else {
                for (int i = 0; i < warmup; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel4x8Bal<int32_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel4x8Bal<int32_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            }
        } else if (tile_m == 8) {
           if (atbsr.useColIdxInt16) {
                for (int i = 0; i < warmup; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel8x4Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel8x4Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            } else {
                for (int i = 0; i < warmup; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel8x4Bal<int32_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    beta_vecY<<<(atbsr.nrows+127)/128, 128>>>(d_y, 1.0, atbsr.nrows);
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel8x4Bal<int32_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            }
        } else if (tile_m == 16) {
           if (atbsr.useColIdxInt16) {
                for (int i = 0; i < warmup; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel16x2Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel16x2Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            } else {
                for (int i = 0; i < warmup; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel16x2Bal<int32_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel16x2Bal<int32_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            }
        } else if (tile_m == 32) {
           if (atbsr.useColIdxInt16) {
                for (int i = 0; i < warmup; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel32x1Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel32x1Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            } else {
                for (int i = 0; i < warmup; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel32x1Bal<int32_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    cudaMemset(d_y, 0, atbsr.nrows * sizeof(double));
                    TcSPMVKernel32x1Bal<int32_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            }
        } else {
            printf("err tile_m = %d\n", tile_m);
        }
    } else {
        if (tile_m == 1) {
            if (atbsr.useColIdxInt16) {
                for (int i = 0; i < warmup; i++) {
                    TcSPMVKernel1x32CC<int16_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    TcSPMVKernel1x32CC<int16_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;

            } else {
                for (int i = 0; i < warmup; i++) {
                    TcSPMVKernel1x32CC<int32_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    TcSPMVKernel1x32CC<int32_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            }
        } else if (tile_m == 2) {
            if (atbsr.useColIdxInt16) {
                for (int i = 0; i < warmup; i++) {
                    TcSPMVKernel2x16CC<int16_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    TcSPMVKernel2x16CC<int16_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;

            } else {
                for (int i = 0; i < warmup; i++) {
                    TcSPMVKernel2x16CC<int32_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    TcSPMVKernel2x16CC<int32_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            }
        } else if (tile_m == 4) {
            if (atbsr.useColIdxInt16) {
                for (int i = 0; i < warmup; i++) {
                    TcSPMVKernel4x8CC<int16_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    TcSPMVKernel4x8CC<int16_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;

            } else {
                for (int i = 0; i < warmup; i++) {
                    TcSPMVKernel4x8CC<int32_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    TcSPMVKernel4x8CC<int32_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            }
        } else if (tile_m == 8) {
            if (atbsr.useColIdxInt16) {
                for (int i = 0; i < warmup; i++) {
                    TcSPMVKernel8x4CC<int16_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    TcSPMVKernel8x4CC<int16_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;

            } else {
                for (int i = 0; i < warmup; i++) {
                    TcSPMVKernel8x4CC<int32_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    TcSPMVKernel8x4CC<int32_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            }
        }  else if (tile_m == 16) {
            if (atbsr.useColIdxInt16) {
                for (int i = 0; i < warmup; i++) {
                    TcSPMVKernel16x2CC<int16_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    TcSPMVKernel16x2CC<int16_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;

            } else {
                for (int i = 0; i < warmup; i++) {
                    TcSPMVKernel16x2CC<int32_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    TcSPMVKernel16x2CC<int32_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            }
        } else if (tile_m == 32) {
            if (atbsr.useColIdxInt16) {
                for (int i = 0; i < warmup; i++) {
                    TcSPMVKernel32x1CC<int16_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    TcSPMVKernel32x1CC<int16_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;

            } else {
                for (int i = 0; i < warmup; i++) {
                    TcSPMVKernel32x1CC<int32_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    TcSPMVKernel32x1CC<int32_t><<<gridSize, blockSize>>>(atbsr.blcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.brow, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            }
        }
    }
    cudaDeviceSynchronize();
    double *originY = (double*)malloc(atbsr.nrows * sizeof(double));
    memset(originY, 0, atbsr.nrows * sizeof(double));
    cudaMemcpy(originY, d_y, atbsr.nrows * sizeof(double), cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());
#ifdef DEBUG
    printf("end atspmv\n");
#endif
    for (int i = 0; i < atbsr.nrows; i++) {
        if (atbsr.recorderCPU[i] >= atbsr.nrows || atbsr.recorderCPU[i] < 0) {
            printf("%d %d\n", i, atbsr.recorderCPU[i]);

        }
        y[atbsr.recorderCPU[i]] = originY[i];
    }
#ifdef DEBUG
    printf("end recorder\n");
#endif
    free(originY);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
void blockInfo(CSRMatrix<double, int32_t>& csr, int TILE_SIZE) {
    int blockCount = 0;
    int tile_m = TILE_SIZE;
    int tile_k = TILE_SIZE;
    for (int i = 0; i < csr.nrows; i += tile_m) {
        int start = i;
        int end = min(i + tile_m, csr.nrows);
        set<int> bset;
        for (int j = start; j < end; j++) {
            for (int z = csr.row_ptr[j]; z < csr.row_ptr[j + 1]; z++) {
                bset.insert(csr.col_idx[z]/tile_k);
            }
        }
        blockCount += bset.size();
    }
    printf(",%f", (double)csr.nnz/(TILE_SIZE*TILE_SIZE*blockCount));
}

// Experimental configurations share the product implementation so packing,
// launch geometry and occupancy cannot diverge between the two public APIs.
static void configuredHostSpmv(CSRMatrix<double, int32_t>& csr, double* x, double* y,
                               int warmup, int repeat, double& time, double& pre,
                               int64_t& bytes, int tile) {
    if (repeat <= 0 || warmup < 0) throw std::invalid_argument("invalid iteration counts");
    thrust::device_vector<int> ptr(csr.row_ptr, csr.row_ptr + csr.nrows + 1);
    thrust::device_vector<int> col(csr.col_idx, csr.col_idx + csr.nnz);
    thrust::device_vector<double> val(csr.val, csr.val + csr.nnz);
    thrust::device_vector<double> dx(x, x + csr.ncols), dy(csr.nrows);
    CSRMatrix<double, int32_t> device{};
    device.nrows = csr.nrows; device.ncols = csr.ncols; device.nnz = csr.nnz;
    device.row_ptr = thrust::raw_pointer_cast(ptr.data());
    device.col_idx = thrust::raw_pointer_cast(col.data());
    device.val = thrust::raw_pointer_cast(val.data());
    AtBsr<double, int32_t> packed;
    auto release = [&] {
        cudaFree(packed.blcVal); cudaFree(packed.blcIdx); cudaFree(packed.blcIdxInt16);
        cudaFree(packed.blcPtr); cudaFree(packed.recorder);
        cudaFree(packed.targetRowByWarp); cudaFree(packed.warpBlcPtr);
    };
    try {
        auto start = std::chrono::steady_clock::now();
        CSRToAtBSRProduct(device, packed, tile);
        pre = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        bytes = int64_t(packed.blockCount) * 32 * (8 + (packed.useColIdxInt16 ? 2 : 4)) +
                int64_t(packed.brow + 1 + packed.nrows) * sizeof(int);
        if (packed.useBalance) bytes += (int64_t(packed.brow) + 2 + 2LL * packed.warpCount) * sizeof(int);
        auto launch = [&] {
            AtSpmvProduct(packed, thrust::raw_pointer_cast(dx.data()),
                          thrust::raw_pointer_cast(dy.data()), 1.0, 0.0);
            atspmv::modelCudaCheck(cudaDeviceSynchronize());
        };
        for (int i = 0; i < warmup; ++i) launch();
        start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeat; ++i) launch();
        time = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / repeat;
        atspmv::modelCudaCheck(cudaMemcpy(y, thrust::raw_pointer_cast(dy.data()),
                                         csr.nrows * sizeof(double), cudaMemcpyDeviceToHost));
    } catch (...) { release(); throw; }
    release();
}

void Spmv(CSRMatrix<double, int32_t>&csrMat, double* x, double* y, int warmup, int repeat, double& avg_time, double& pre_time,int64_t& buffsize, int idx=-1) {
    if (idx < -1 || idx > 5) throw std::invalid_argument("invalid forced tile");
    if (csrMat.nnz == 0 || csrMat.nrows == 0) {
        std::fill(y, y + csrMat.nrows, 0.0);
        avg_time = pre_time = 0;
        buffsize = 0;
        return;
    }
    if (!atspmv::kernelConfigFromEnvironment().baseline()) {
        configuredHostSpmv(csrMat, x, y, warmup, repeat, avg_time, pre_time, buffsize, idx);
        return;
    }
    int minLen = INT32_MAX;
    int maxLen = 0;
    int nonzeroRows = 0;
    for (int i = 0; i < csrMat.nrows; i++) {
        int len = csrMat.row_ptr[i+1] - csrMat.row_ptr[i];
        if (len == 0) continue;
        nonzeroRows++;
        minLen = min(minLen, len);
        maxLen = max(maxLen, len);
    }
    csrMat.maxLen = maxLen;
    csrMat.minLen = minLen;
    double stand = 0;
    double avg_nnz = double(csrMat.nnz) / nonzeroRows;
    for (int i = 0; i < csrMat.nrows; i++) {
        int len = csrMat.row_ptr[i + 1] - csrMat.row_ptr[i];
        if (len == 0) continue;
        stand += (len - avg_nnz) * (len - avg_nnz);
    }
    stand = sqrtf(stand / nonzeroRows);
    csrMat.stand = stand;
    csrMat.avg_nnz = avg_nnz;
    csrMat.nonzeroRows = nonzeroRows;
    int maxAvgRatio = maxLen / avg_nnz;
    csrMat.useFixLen = minLen == maxLen && nonzeroRows == csrMat.nrows;
    csrMat.FixLen = csrMat.useFixLen ? maxLen : 0;
    csrMat.useColInt16 = csrMat.ncols < INT16_MAX;
    if (idx == -1 && stand < 12 && avg_nnz < 8 && maxAvgRatio < 4) {
        pre_time = 0;
        buffsize = 0;
        spmv_fp64_warpper(csrMat, x, y, warmup, repeat, avg_time);
        return;
    }

    // CSR2A
    AtBsr<double, int32_t> atBsr;
    CSRToAtBSR(csrMat, atBsr, pre_time, idx, buffsize);
    CHECK(cudaGetLastError());
    // atBsr.useBalance = false;
    AtSpmv(atBsr, x, y, warmup, repeat, avg_time);
    cudaFree(atBsr.blcIdx);
    cudaFree(atBsr.blcIdxInt16);
    cudaFree(atBsr.blcVal);
    cudaFree(atBsr.blcPtr);
    cudaFree(atBsr.recorder);
    cudaFree(atBsr.targetRowByWarp);
    cudaFree(atBsr.warpBlcPtr);
    free(atBsr.recorderCPU);
    cudaDeviceSynchronize();
}
