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
// #define DEBUG 

__global__ void getTileDensityAndAtomKernel(int *row_len, int nrow, int* blockcount,int *atomTimes, int cap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nrow) {
        int block =  (row_len[idx] + 31)/32;
        atomicAdd(&blockcount[0], block);
        atomicAdd(&atomTimes[0], (block + cap - 1) / cap);
    }

    if (idx*2 + 1 < (nrow+1)/2*2 ) {
        int targetidx = min(idx*2+1, nrow-1);
        int block =  (row_len[targetidx] + 15)/16;
        atomicAdd(&blockcount[1], block);
        atomicAdd(&atomTimes[1], (block + cap - 1) / cap);
    }
    if (idx*4 + 3 < (nrow+3)/4*4 ) {
        int targetidx = min(idx*4+3, nrow-1);
        int block =  (row_len[targetidx] + 7)/8;

        atomicAdd(&blockcount[2], block);
        atomicAdd(&atomTimes[2], (block + cap - 1) / cap);
    }

    if (idx*8 + 7 < (nrow+7)/8*8 ) {
        int targetidx = min(idx*8+7, nrow-1);
        int block = (row_len[targetidx]+3)/4;

        atomicAdd(&blockcount[3], block);
        atomicAdd(&atomTimes[3], (block + cap - 1) / cap);
    }

    if (idx*16 + 15 < (nrow+15)/16*16 ) {
        int targetidx = min(idx*16+15, nrow-1);
        int block = (row_len[targetidx]+1)/2;

        atomicAdd(&blockcount[4], block);
        atomicAdd(&atomTimes[4], (block + cap - 1) / cap);
    }

    if (idx*32 + 31 < (nrow+31)/32*32 ) {
        int targetidx = min(idx*32+31, nrow-1);
        int block = row_len[targetidx];
        // printf("block = %d\n", block);
        atomicAdd(&blockcount[5], block);
        atomicAdd(&atomTimes[5], (block + cap - 1) / cap);
    }
}
struct RowNNZCalculator {
    const int32_t* row_ptr;
    
    RowNNZCalculator(const int32_t* _row_ptr) : row_ptr(_row_ptr) {}
    
    __host__ __device__
    int32_t operator()(int32_t row_index) const {
        return row_ptr[row_index + 1] - row_ptr[row_index];
    }
};

template<typename T>
struct sq_dev {
    const double mean;
    sq_dev(double m) : mean(m) {}
    __host__ __device__ double operator()(const T& x) const {
        double d = static_cast<double>(x) - mean;
        return d * d;
    }
};

int getMaxPhyThread(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    size_t total_concurrent_threads = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
    return total_concurrent_threads;
}
pair<int, bool> selectTile(double scores[6], int noBalWarpCounts[6], int balWarpCounts[6], double MaxScore, int pyhThreads,int avg_nnz,bool bal) {
    int targetTile = -1;
    for (int i = 0; i < 6; i++) {
        if (scores[i] == MaxScore) {
            targetTile = i;
            break;
        }
    }
    int warp0 = bal ? balWarpCounts[0] : noBalWarpCounts[0];
    if (warp0*32 <= pyhThreads) {
        targetTile = 0;
    }
    int thread = bal?  balWarpCounts[targetTile] : noBalWarpCounts[targetTile];
    int tile_m = 1<<targetTile;
    int tile_k = 32/tile_m;
    bool useBalance = false;
    if (thread*32 < pyhThreads && avg_nnz > tile_k * 12) {
        useBalance = true;
    } else {
        useBalance = bal;
    }
    return make_pair(targetTile, useBalance);
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
        int warpID = warpPtr[rowid] + j ;
        warpBlcPtr[warpPtr[rowid] + j] = min(end, start + j * 8);
    }
}
template<typename IDX_TYPE>
__global__ void csr2AtBsrKernel(double* d_val,int *rowInx, int* d_colIdx,int* drow_ptr,int brow,int nrows,int tile_m, int tile_k, int* blc_ptr,double* blc_val, IDX_TYPE *blc_idx) {
    int rowid = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowid >= nrows) return;
    int originrowid = rowid;
    rowid = rowInx[rowid];
    // if (rowid == 32) {
    //     printf("rowid %d %d\n", rowid, originrowid);
    // }
    int bid = originrowid/tile_m;
    int pre = blc_ptr[bid];
    // if (rowid == 32) {
    //     for (int i = drow_ptr[rowid], j = 0; i < drow_ptr[rowid+1]; i++,j++) {
    //         printf("d_val %d %f\n", i, d_val[i]);
    //     }
    // }
    for (int i = drow_ptr[rowid], j = 0; i < drow_ptr[rowid+1]; i++,j++) {
        int innerbid = j/tile_k;
        int inneridx = (originrowid%tile_m)*tile_k + j%tile_k;
        blc_val[(pre + innerbid)*32 + inneridx] = d_val[i];
        blc_idx[(pre + innerbid)*32 + inneridx] = IDX_TYPE(d_colIdx[i]);
    }
    // if (rowid == 32) {
    //     for (int i = pre*32;i < pre*32+32;i++) {
    //         printf("blc_val %d %f\n", i, blc_val[i]);
    //     }
    // }
}
// 354

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
    // if (targetRow == 0) {
    //     printf("warpId = %d, warpBlcPtr = %d warpBlcPtr+1=%d\n", warpId, warpBlcPtr[warpId],warpBlcPtr[warpId+1]);
    // }
    int j = (warpBlcPtr[warpId] - blcPtr[targetRow/tile_m])*tile_k + laneId%tile_k;
    for (int i = warpBlcPtr[warpId]; i < warpBlcPtr[warpId+1]; i++,j += tile_k) {
        int csridx = csrStartIdx + j;
        int bsridx = i*32 + laneId;
        // if (csridx < csrEndIdx && targetRow == 0) {
        //     printf("val: %f\n", d_val[csridx]);
        // }
        blc_val[bsridx] = csridx < csrEndIdx ? d_val[csridx]: 0;
        blc_idx[bsridx] = csridx < csrEndIdx ? IDX_TYPE(d_colIdx[csridx]) :-1;
    }
}

__host__ int CSRToAtBSR(CSRMatrix<double, int32_t>&csrMat, AtBsr<double, int32_t>&atbsr, double& preTime, int idx) {
    // check use int16
    double *d_val;
    int *d_colIdx;
    int *d_rowPtr;
#ifdef DEBUG
    // printf("nrows = %d, nonzeroRows = %d\n", csrMat.nrows, csrMat.ncols)
#endif

    CHECK(cudaMalloc((void**)&d_val, csrMat.nnz * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_colIdx, csrMat.nnz * sizeof(int)));
    CHECK(cudaMalloc((void**)&d_rowPtr, (csrMat.nrows + 1) * sizeof(int)));
    CHECK(cudaMemcpy(d_val, csrMat.val, csrMat.nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_colIdx, csrMat.col_idx, csrMat.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_rowPtr, csrMat.row_ptr, (csrMat.nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    auto start = std::chrono::high_resolution_clock::now();

    // TODO
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
    thrust::sort(thrust::device, drow_len, drow_len + csrMat.nrows);
#ifdef DEBUG
    printf("sort done\n");
#endif
    // int maxLen = -1, minLen = INT32_MAX;
    // find the best tile size
    int *d_blockcount;
    int *d_warpCount;
    int *h_blockcount;
    int *h_warpCount;
    h_warpCount = (int*)malloc(6 * sizeof(int));
    h_blockcount = (int*)malloc(6 * sizeof(int));
    cudaMalloc(&d_blockcount, 6 * sizeof(int));
    cudaMemset(d_blockcount, 0, 6 * sizeof(int));
    cudaMalloc(&d_warpCount, 6 * sizeof(int));
    cudaMemset(d_warpCount, 0, 6 * sizeof(int));

    getTileDensityAndAtomKernel<<<(csrMat.nrows + 256 - 1)/256, 256>>>(drow_len, csrMat.nrows, d_blockcount, d_warpCount, 8);
    cudaDeviceSynchronize();
    cudaMemcpy(h_blockcount, d_blockcount, 6 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_warpCount, d_warpCount, 6 * sizeof(int), cudaMemcpyDeviceToHost);

    // double avg_nnz = double(csrMat.nnz)/csrMat.nrows;
    // double sq_sum = thrust::transform_reduce(thrust::device, drow_len + csrMat.nrows, drow_len + csrMat.nrows, sq_dev<int>(avg_nnz),
    //                                       0.0, thrust::plus<double>());
    // double stand = sqrt(sq_sum/csrMat.nrows);
    if (csrMat.stand > 12 || csrMat.maxLen/csrMat.avg_nnz > 2) {
        atbsr.useBalance = 1;
    }
    int pyhThreads = getMaxPhyThread(0);
    double densityArr[6], score[6], MaxScore = -1;
    int balWarpCounts[6];
    int noBalWarpCounts[6];
    for (int i = 0; i < 6; i++) {
        balWarpCounts[i] = h_warpCount[i];
        noBalWarpCounts[i] = (csrMat.nrows + (1<<i) - 1)/(1<<i);
        // printf("bal: %d noBal: %d\n", balWarpCounts[i], noBalWarpCounts[i]);
    }

    int m = 1;
    for (int i = 0; i < 6; i++) {
        // printf("tile %d block count = %d\n", i, h_blockcount[i]);
        densityArr[i] = double(csrMat.nnz)/(h_blockcount[i]*32);
        // printf("tile %d density = %.2f\n", i, densityArr[i]);
        score[i] = densityArr[i]*0.9 - 0.1*double(balWarpCounts[i]*m/csrMat.nnz * atbsr.useBalance) - 0.01*(double(h_blockcount[i]*(5-i))/csrMat.nnz);
        // printf("GPU %f %d %f h_blockcount:%d\n",densityArr[i], h_atomTimes[i]*m, score[i],h_blockcount[i]);
        MaxScore = max(MaxScore, score[i]);
        m *= 2;
    }

    int thread = 0;
    int warpCount;
    int blockCount;
    // int atomTimes;
    pair<int,bool>res = selectTile(score, noBalWarpCounts, balWarpCounts, MaxScore, pyhThreads, csrMat.avg_nnz, atbsr.useBalance);
#ifdef DEBUG
    printf("selectTile done\n");
#endif
    if (idx != -1) {
        res.first = idx;
        printf("den:%f score:%f\n", densityArr[idx],score[idx]);
    } else {
        printf("select: %d dens:%f  score:%f\n", res.first, densityArr[res.first],score[res.first]);
    }
    int tile_m = 1<<(res.first);

    int tile_k = 32/tile_m;
    atbsr.useBalance = res.second;
    warpCount = atbsr.useBalance ? balWarpCounts[res.first] : noBalWarpCounts[res.first]; 
    blockCount = h_blockcount[res.first];
    // convert csr to atbsr
    int *blc_ptr;
    int brow = (csrMat.nrows + tile_m - 1)/tile_m;
    cudaMalloc(&blc_ptr, (brow+1)*sizeof(int));
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

    getBlcPtrKernel<<<(brow + 255)/256 ,256>>>(drow_len, tile_m, tile_k, csrMat.nrows, blc_ptr);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, blc_ptr, blc_ptr + (brow + 1), blc_ptr, 0);
    // int *h_blcPtr = (int*)malloc((brow + 1)*sizeof(int));
    // cudaMemcpy(h_blcPtr, blc_ptr, (brow + 1)*sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < brow; i++) {
    //     printf("blc_ptr[%d]:%d\n", i, h_blcPtr[i]);
    // }
    int *warpBlcPtr;
    int *warpTargetRow;
    int *rowPtrbyWarp;
    int *rowIdxbyWarp;

    cudaMalloc(&rowPtrbyWarp, (brow+1)*sizeof(int));
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
        // if (atbsr.useColIdxInt16)
        //     csr2AtBsrKernel<int16_t><<<(csrMat.nrows + 255)/256 ,256>>>(d_val, drow_index, d_colIdx,  d_rowPtr, brow, csrMat.nrows,tile_m,tile_k, blc_ptr,blc_val,blc_idxint16);
        // else
        //     csr2AtBsrKernel<int32_t><<<(csrMat.nrows + 255)/256 ,256>>>(d_val, drow_index, d_colIdx,  d_rowPtr, brow, csrMat.nrows,tile_m,tile_k, blc_ptr,blc_val,blc_idx);
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
    // int *drowLen;
    // printf("atbsr preprocess time = %d ms\n", duration.count());

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

    return 0;
}
#define WARP_CAPACITY 32

template <typename IDX_TYPE>
__global__ void spmv_fp64(int32_t *rowPtr, IDX_TYPE *colIdx, int32_t rows, int32_t nnz, double* values,double* x, double* outvalues) { 
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= rows) return;
    double tmp = 0.0;
    // printf("rowPtr[tid]:%d rowPtr[tid+1]:%d\n", rowPtr[tid], rowPtr[tid+1]);
    // int count = rowPtr[tid+1] - rowPtr[tid];
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
    // int count = rowPtr[tid+1] - rowPtr[tid];
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
    // printf("useColInt16:%d %d", csr.useColInt16,csr.useFixLen);
    if (!csr.useColInt16) {
        // printf("use Int32\n");
        if (csr.useFixLen) {
            // printf("use fixlen\n");

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
    // if (laneId == 0 && warpId == 50) {
    //     printf("warpBlcPtr %d %d targetRow %d\n", warpBlcPtr[warpId], warpBlcPtr[warpId + 1], warpTargetRow[warpId]);
    // }
    // #pragma unroll 4
    // if (laneId == 0) {
    //     printf("BlcPtr %d %d\n", BlcPtr[warpId], BlcPtr[warpId + 1]);
    // }
    for (int i = BlcPtr[warpId]; i < BlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;

    }
    // printf("fragA %f fragB %f fragC %f\n", fragA, fragB, fragC);
    // int warpRowId = warpTargetRow[warpId];
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
    // if (laneId == 0 && warpId == 50) {
    //     printf("warpBlcPtr %d %d targetRow %d\n", warpBlcPtr[warpId], warpBlcPtr[warpId + 1], warpTargetRow[warpId]);
    // }
    // #pragma unroll 4
    // if (laneId == 0) {
    //     printf("BlcPtr %d %d\n", BlcPtr[warpId], BlcPtr[warpId + 1]);
    // }
    for (int i = BlcPtr[warpId]; i < BlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;

    }
    // printf("fragA %f fragB %f fragC %f\n", fragA, fragB, fragC);
    // int warpRowId = warpTargetRow[warpId];
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
    // if (laneId == 0 && warpId == 50) {
    //     printf("warpBlcPtr %d %d targetRow %d\n", warpBlcPtr[warpId], warpBlcPtr[warpId + 1], warpTargetRow[warpId]);
    // }
    // #pragma unroll 4
    // if (laneId == 0) {
    //     printf("BlcPtr %d %d\n", BlcPtr[warpId], BlcPtr[warpId + 1]);
    // }
    for (int i = BlcPtr[warpId]; i < BlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        // if (colIdx < -1) {
            // printf("colIdx %d\n", colIdx);
        // }
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;

    }
    // int warpRowId = warpTargetRow[warpId];
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
    // if (laneId == 0) {
    //     printf("warpID: %d  blockCount: %d\n", warpId, blcPtr[warpId + 1] - blcPtr[warpId]);
    // }
    #pragma unroll 4
    for (int i = blcPtr[warpId]; i < blcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;
    }
        // if (warpId == 0)

    // printf("colIdx %d fragA %f fragB %f fragC %f\n", 1, fragA, fragB, fragC);

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
    // if (laneId == 0 && warpId == 50) {
    //     printf("warpBlcPtr %d %d targetRow %d\n", warpBlcPtr[warpId], warpBlcPtr[warpId + 1], warpTargetRow[warpId]);
    // }
    // #pragma unroll 4
    // if (laneId == 0) {
    //     printf("BlcPtr %d %d\n", BlcPtr[warpId], BlcPtr[warpId + 1]);
    // }
    for (int i = BlcPtr[warpId]; i < BlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;

    }
    // printf("fragA %f fragB %f fragC %f\n", fragA, fragB, fragC);
    // int warpRowId = warpTargetRow[warpId];
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
        // if (rowId == 0)
        // printf("i:%d fragA:%f fragB %f fragC:%f tid:%d idx:%d\n",i, fragA, fragB,fragC, tid, i*32 + laneId);
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
    // if (laneId == 0 && warpId == 50) {
    //     printf("warpBlcPtr %d %d targetRow %d\n", warpBlcPtr[warpId], warpBlcPtr[warpId + 1], warpTargetRow[warpId]);
    // }
    // #pragma unroll 4

    for (int i = BlcPtr[warpId]; i < BlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;

    }
    // printf("fragA %f fragB %f fragC %f\n", fragA, fragB, fragC);
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
    // printf("fragA %f fragB %f fragC %f\n", fragA, fragB, fragC);
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

int AtSpmv(AtBsr<double, int32_t>& atbsr, double* x, double* y, int warmup, int repeat, double &avgTime) {

    int tile_m = atbsr.TILE_M;
    int tile_k = atbsr.TILE_K;
#ifdef DEBUG
    printf("AtSpmv\n");
    printf("tile_m %d tile_k %d bal: %d int16:%d\n", tile_m, tile_k, atbsr.useBalance, atbsr.useColIdxInt16);
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
                    TcSPMVKernel1x32Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    TcSPMVKernel1x32Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                
                auto end = std::chrono::high_resolution_clock::now(); 
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            } else {
                for (int i = 0; i < warmup; i++) {
                    TcSPMVKernel1x32Bal<int32_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
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
                    TcSPMVKernel2x16Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    TcSPMVKernel2x16Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                
                auto end = std::chrono::high_resolution_clock::now(); 
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            } else {
                for (int i = 0; i < warmup; i++) {
                    TcSPMVKernel2x16Bal<int32_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
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
                    TcSPMVKernel4x8Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    TcSPMVKernel4x8Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                
                auto end = std::chrono::high_resolution_clock::now(); 
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            } else {
                for (int i = 0; i < warmup; i++) {
                    TcSPMVKernel4x8Bal<int32_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
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
                    TcSPMVKernel8x4Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    TcSPMVKernel8x4Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                
                auto end = std::chrono::high_resolution_clock::now(); 
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            } else {
                for (int i = 0; i < warmup; i++) {
                    TcSPMVKernel8x4Bal<int32_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
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
                    TcSPMVKernel16x2Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    TcSPMVKernel16x2Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                
                auto end = std::chrono::high_resolution_clock::now(); 
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            } else {
                for (int i = 0; i < warmup; i++) {
                    TcSPMVKernel16x2Bal<int32_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
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
                    TcSPMVKernel32x1Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
                    TcSPMVKernel32x1Bal<int16_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdxInt16, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                
                auto end = std::chrono::high_resolution_clock::now(); 
                auto elapsed = end - start;
                avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
            } else {
                for (int i = 0; i < warmup; i++) {
                    TcSPMVKernel32x1Bal<int32_t><<<gridSize, blockSize>>>(atbsr.targetRowByWarp ,atbsr.warpBlcPtr, atbsr.blcIdx, atbsr.blcVal, atbsr.warpCount, atbsr.nrows, d_x, d_y);
                    cudaDeviceSynchronize();
                }
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < repeat; i++) {
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
            for (int z = csr.row_ptr[j], y = 0; z < csr.row_ptr[j + 1]; z++) {
                bset.insert(csr.col_idx[z]/tile_k);
            }
        }
        blockCount += bset.size();
    }
    printf(",%f", (double)csr.nnz/(TILE_SIZE*TILE_SIZE*blockCount));
}

void Spmv(CSRMatrix<double, int32_t>&csrMat, double* x, double* y, int warmup, int repeat, double& avg_time, double& pre_time, int idx=-1) {
    int minLen;
    int maxLen;
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
    // printf("now\n");
    for (int i = 0; i < nonzeroRows; i++) {
        int len = csrMat.row_ptr[i + 1] - csrMat.row_ptr[i];
        if (len == 0) continue;
        stand += (len - avg_nnz) * (len - avg_nnz);
    }
    stand = sqrtf(stand / nonzeroRows);
    csrMat.stand = stand;
    csrMat.avg_nnz = avg_nnz;
    csrMat.nonzeroRows = nonzeroRows;
    int maxAvgRatio = maxLen / avg_nnz;
    if (minLen == maxLen && nonzeroRows == csrMat.nrows) {
        csrMat.FixLen = true;
    }
    if (csrMat.ncols < INT16_MAX) {
        csrMat.useColInt16 = true;
    }
    // csrMat.useColInt16 = false;
    // printf("stand %f avg_nnz %f maxAvgRatio %d\n", stand, avg_nnz, maxAvgRatio);
    if (stand < 12 && avg_nnz < 8 && maxAvgRatio < 4) {
        // printf("-1,-1,-1,-1,-1,");
        spmv_fp64_warpper(csrMat, x, y, warmup, repeat, avg_time);
        return;
    }
        csrMat.useColInt16 = true;

    // CSR2A
    AtBsr<double, int32_t> atBsr;
    CSRToAtBSR(csrMat, atBsr, pre_time, idx);
    // for (int i = 2;i <= 8;i*=2) {
    //     blockInfo(csrMat, i);
    // }
    //     // printf("\n");
    // return;
    CHECK(cudaGetLastError());
    // atBsr.useBalance = false;
    AtSpmv(atBsr, x, y, warmup, repeat, avg_time);
    cudaFree(atBsr.blcIdx);
    cudaFree(atBsr.blcIdxInt16);
    cudaFree(atBsr.blcVal);
    cudaFree(atBsr.blcPtr);
    cudaDeviceSynchronize();
}