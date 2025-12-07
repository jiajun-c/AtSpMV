#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include "csrReader.h"
#include "common.h"

// #define CHECK_CUDA(call) \
//     if ((call) != cudaSuccess) { \
//         fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(call)); exit(1); }

// #define CHECK_CUSPARSE(call) \
//     if ((call) != CUSPARSE_STATUS_SUCCESS) { \
//         fprintf(stderr, "cuSPARSE error at %s:%d\n", __FILE__, __LINE__); exit(1); }

__global__ void jacobi_update(
    int n, const double* diag, const double* b,
    const double* Ax, double* x_new, const double* x_old)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double r = b[i] - (Ax[i] - diag[i] * x_old[i]);
        x_new[i] = r / diag[i];
    }
}

__global__ void compute_residual(
    int n, const double* b, const double* Ax, double* r)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) r[i] = b[i] - Ax[i];
}

__global__ void reduce_l2(int n, const double* r, double* out)
{
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    double v = (i < n) ? r[i] * r[i] : 0.0;
    sdata[tid] = v;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

int main(int argc, char** argv)
{
    // if (argc < 3) {
    //     std::cerr << "Usage: ./jacobi A.mtx b.mtx\n";
    //     return 1;
    // }

    CSRMatrix<double,int32_t> csr = read_mtx(argv[1]);
    // CSRMatrix<double,int32_t> rhs = read_mtx(argv[2]);

    int n = csr.nrows;
    int nnz = csr.nnz;

    int* h_rowptr = csr.row_ptr;
    int* h_colidx = csr.col_idx;
    double* h_vals = csr.val;
    double* h_b = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) h_b[i] = 1.0;

    // --- Extract diagonal ---
    std::vector<double> h_diag(n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = h_rowptr[i]; j < h_rowptr[i + 1]; j++) {
            if (h_colidx[j] == i) {
                h_diag[i] = h_vals[j];
            }
        }
        if (h_diag[i] == 0) {
            std::cerr << "Matrix has zero diagonal at row " << i << std::endl;
            exit(1);
        }
    }

    // Allocate device memory
    double *d_x_old, *d_x_new, *d_b, *d_diag, *d_Ax, *d_r;
    int *d_rowptr, *d_colidx;
    double *d_vals;

    CHECK_CUDA(cudaMalloc(&d_rowptr, (n+1)*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_colidx, nnz*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_vals, nnz*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_b, n*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_diag, n*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_x_old, n*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_x_new, n*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_Ax, n*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_r, n*sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_rowptr, h_rowptr, (n+1)*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colidx, h_colidx, nnz*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, h_vals, nnz*sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, n*sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_diag, h_diag.data(), n*sizeof(double), cudaMemcpyHostToDevice));

    std::vector<double> init_x(n, 0.0);
    CHECK_CUDA(cudaMemcpy(d_x_old, init_x.data(), n*sizeof(double), cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, n, n, nnz,
        d_rowptr, d_colidx, d_vals,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    cusparseDnVecDescr_t vecX, vecY;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, d_x_old, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, n, d_Ax, CUDA_R_64F));

    size_t bufferSize;
    void* dBuffer = nullptr;
    double alpha = 1.0, beta = 0.0;

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // --- Jacobi iteration ---
    const int max_iter = 100000;
    const double tol = 1e-4;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    std::vector<double> h_partial(blocks);
    double *d_partial, *d_resnorm;
    CHECK_CUDA(cudaMalloc(&d_partial, blocks * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_resnorm, sizeof(double)));

    for (int k = 0; k < max_iter; k++) {

        // Ax_old = A * x_old
        CHECK_CUSPARSE(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

        // Jacobi: x_new = (b - (Ax - diag*x_old)) / diag
        jacobi_update<<<blocks, threads>>>(n, d_diag, d_b, d_Ax, d_x_new, d_x_old);

        // Swap pointers
        std::swap(d_x_old, d_x_new);
        CHECK_CUSPARSE(cusparseDnVecSetValues(vecX, d_x_old));

        // Compute Ax_new = A * x_old

        if (k%100 == 0) {
            CHECK_CUSPARSE(cusparseSpMV(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        // r = b - Ax_new
            compute_residual<<<blocks, threads>>>(n, d_b, d_Ax, d_r);

            // L2 norm of residual
            reduce_l2<<<blocks, threads>>>(n, d_r, d_partial);
            reduce_l2<<<1, threads>>>(blocks, d_partial, d_resnorm);

            double resnorm;
            CHECK_CUDA(cudaMemcpy(&resnorm, d_resnorm, sizeof(double), cudaMemcpyDeviceToHost));
            resnorm = sqrt(resnorm);

            std::cout << "Iter " << k << " residual = " << resnorm << "\n";

            if (resnorm < tol) {
                std::cout << "Converged at iter " << k << "\n";
                break;
            }
        }
    }

    // Copy result back
    std::vector<double> x(n);
    CHECK_CUDA(cudaMemcpy(x.data(), d_x_old, n*sizeof(double), cudaMemcpyDeviceToHost));

    std::cout << "Final x[0] = " << x[0] << " ...\n";

    return 0;
}
