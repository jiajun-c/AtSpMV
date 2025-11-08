#include "csrspmv.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include "matrixFormat.h"
using namespace std;

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

__global__ void spmv_fp64_balance(int32_t* rowPtr, int32_t *rowPtrbyWarp, int32_t *colIdx, int32_t* rowIdxbyWarp, int32_t rows, int32_t nnz, int32_t warp_num, double* values, double* x, double* d_y) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= warp_num)
        return;
    int target_row_id = rowIdxbyWarp[tid];
    if (target_row_id >= rows) {
        return;
    }
    int start = rowPtr[target_row_id] + (tid - rowPtrbyWarp[target_row_id]) * WARP_CAPACITY;
    int end = start + WARP_CAPACITY < rowPtr[target_row_id + 1] ? start + WARP_CAPACITY : rowPtr[target_row_id + 1];
    double res = 0;
    // printf("start:%d end:%d\n", start, end );
    #pragma unroll 4
    for (int i = start; i < end; i++) {
        res += values[i] * x[colIdx[i]];
    }
    atomicAdd(&d_y[target_row_id],res);
}

__global__ void get_rowPtrbyWarp_csr(int *d_blcPtr, int *rowPtrbyWarp, int blc_row)
{
    int rowid = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowid >= blc_row)
        return;
    rowPtrbyWarp[rowid] = (d_blcPtr[rowid + 1] - d_blcPtr[rowid] + WARP_CAPACITY - 1) / WARP_CAPACITY;
}


__global__ void get_rowIdxbyWarp_csr(int *rowPtrbyWarp, int *rowIdxbyWarp, int blc_row)
{
    int rowid = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowid >= blc_row)
        return;

    int offset = rowPtrbyWarp[rowid];
    int stride = rowPtrbyWarp[rowid + 1] - rowPtrbyWarp[rowid];

    for (int i = offset; i < (offset + stride); i++)
    {
        rowIdxbyWarp[i] = rowid;
    }
}

int spmv_fp64_warpper(CSRMatrix<double, int32_t>&csr ,double* x, double* outvalues, int warmup, int repeat, double& avg_time) {
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

int spmv_fp64_balance_warpper(CSRMatrix<double, int32_t>&csr ,double* x, double* outvalues, int warmup, int repeat, double& avg_time) {  
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
    // printf("rows:%d cols:%d nnz:%d\n", rows, cols, nnz);

    int32_t *d_rowPtr, *d_colIdx, *rowIdxbyWarp;
    cudaMalloc((void**)&d_rowPtr, sizeof(int32_t)*(rows+1));
    cudaMalloc((void**)&d_colIdx, sizeof(int32_t)*nnz);
    cudaMemcpy(d_rowPtr, csr.row_ptr, sizeof(int32_t)*(rows+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, csr.col_idx, sizeof(int32_t)*nnz, cudaMemcpyHostToDevice);
    int *rowPtrbyWarp;
    cudaMalloc((void**)&rowPtrbyWarp, sizeof(int)*(rows+1));
    int ThreadNum = 32 * 4;
    int BlockNum = (rows + ThreadNum - 1) / ThreadNum;
    get_rowPtrbyWarp_csr<<<BlockNum, ThreadNum>>>(d_rowPtr, rowPtrbyWarp, rows);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, rowPtrbyWarp, rowPtrbyWarp + rows + 1, rowPtrbyWarp, 0);
    cudaDeviceSynchronize();
    int warpnum = 0;
    cudaMemcpy(&warpnum, (rowPtrbyWarp) + rows, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaMalloc((void **)&rowIdxbyWarp, sizeof(int) * warpnum);

    get_rowIdxbyWarp_csr<<<BlockNum, ThreadNum>>>(rowPtrbyWarp, rowIdxbyWarp, rows);
    int threadsNum = 4*32;
    int blocksNum = (warpnum+threadsNum-1)/threadsNum;
    // printf("warpnum: %d threadsNum:%d blocksNum:%d\n", warpnum, threadsNum, blocksNum);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    for (int i = 0; i < warmup; i++) {
            // spmv_fp64<<<blocksNum, threadsNum>>>(d_rowPtr, d_colIdx, rows, nnz, d_values, d_x, d_outvalues);
        spmv_fp64_balance<<<blocksNum, threadsNum>>>(d_rowPtr, rowPtrbyWarp, d_colIdx, rowIdxbyWarp, rows, nnz, warpnum, d_values, d_x, d_outvalues);
        // spmv_fp64_balance<<<blocksNum, threadsNum>>>(d_rowPtr, rowPtrbyWarp, d_colIdx, rowIdxbyWarp, rows, nnz, warpnum, d_values, d_x, d_outvalues);

        cudaDeviceSynchronize();
    }
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; i++) {
        spmv_fp64_balance<<<blocksNum, threadsNum>>>(d_rowPtr, rowPtrbyWarp, d_colIdx, rowIdxbyWarp, rows, nnz, warpnum, d_values, d_x, d_outvalues);
        // spmv_fp64_balance<<<blocksNum, threadsNum>>>(d_rowPtr, rowPtrbyWarp, d_colIdx, rowIdxbyWarp, rows, nnz, warpnum, d_values, d_x, d_outvalues);
        // spmv_fp64<<<blocksNum, threadsNum>>>(d_rowPtr, d_colIdx, rows, nnz, d_values, d_x, d_outvalues);

        cudaDeviceSynchronize();

    }
    auto end = chrono::high_resolution_clock::now();
    auto eplapsed = end - start;
    avg_time = chrono::duration_cast<chrono::microseconds>(eplapsed).count() / repeat;
    CHECK(cudaGetLastError());

    cudaMemcpy(outvalues, d_outvalues, sizeof(double)*rows, cudaMemcpyDeviceToHost);
    cudaFree(rowPtrbyWarp);
    cudaFree(rowIdxbyWarp);
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_outvalues);
    return 0;
}
