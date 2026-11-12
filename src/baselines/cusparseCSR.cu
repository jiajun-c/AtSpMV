#include "cusp.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cusparse.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <cuda_fp16.h>
#include <iostream>
#include "matrixFormat.h"
#include <cusparse_v2.h> 


int cusparse_spmvfp64(CSRMatrix<double, int32_t> &A, double *hX, double *hY, int repeat, double &avg_time, double& pre_time, int64_t &buffsize) {
    double     alpha           = 1.0f;
    double     beta            = 0.0f;
    int32_t   *dA_csrOffsets, *dA_columns;
    double *dA_values, *dX, *dY;
    int A_num_rows = A.nrows;
    int A_nnz = A.nnz;
    int A_num_cols = A.ncols;
    // int32_t* hrow_ptr64 = (int32_t*)malloc((A_num_rows + 1) * sizeof(int32_t));
    // for (int i = 0; i <= A_num_rows; i++) {
    //     hrow_ptr64[i] = A.row_ptr[i];
    // }

    // int32_t* hAcol = (int32_t*)malloc(A_nnz * sizeof(int32_t));
    // for (int i = 0; i < A_nnz; i++) {
    //     hAcol[i] = A.col_idx[i];
    // }
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int32_t)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int32_t))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(double))      )
    CHECK_CUDA( cudaMalloc((void**) &dX,         A_num_cols * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &dY,         A_num_rows * sizeof(double)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, A.row_ptr,
                           (A_num_rows + 1) * sizeof(int32_t),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, A.col_idx, A_nnz * sizeof(int32_t),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, A.val, A_nnz * sizeof(double),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX, hX, A_num_cols * sizeof(double),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dY, hY, A_num_rows * sizeof(double),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    auto startpre = std::chrono::high_resolution_clock::now();
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )

    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_64F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_64F) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) )
    // printf("cusparse buffer size = %d\n", bufferSize);
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    int warmup = 10;
    auto endpre = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endpre - startpre);
    pre_time =  duration.count();
    buffsize =  bufferSize +A.nnz*(sizeof(double) + sizeof(int32_t))+ (A.nrows+1)*sizeof(int);
    // printf("cusparse preprocess time = %d ms\n", duration.count());
    for (int i = 0; i < warmup; i++) {

        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
        cudaDeviceSynchronize();

    }
    auto start = std::chrono::high_resolution_clock::now(); 
    for (int i = 0; i < repeat; i++) {

        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
        cudaDeviceSynchronize();

    }
    auto end = std::chrono::high_resolution_clock::now(); 
    auto elapsed = end - start;
    avg_time = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;

    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hY, dY, A_num_rows * sizeof(double),cudaMemcpyDeviceToHost) )
    return 0;
}
