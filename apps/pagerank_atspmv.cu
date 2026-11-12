#include <cusparse.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "csrReader.h"
#include "atspmvprod.h"
#include <chrono>
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while (0)

#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t err = call; \
        if (err != CUSPARSE_STATUS_SUCCESS) { \
            std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

// CUDA kernel: y[i] = r[i] / out_degree[i]
__global__ void compute_y(const double* __restrict__ r, const double* __restrict__ inv_deg, double* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = r[i] * inv_deg[i];
    }
}

// CUDA kernel: r[i] = alpha * r[i] + (1 - alpha) / n
__global__ void pagerank_update(double* r, double alpha, double inv_n, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        r[i] = alpha * r[i] + inv_n;
    }
}

// CUDA kernel: compute L1 norm of |r_new - r_old|
__global__ void compute_diff(const double* r_old, const double* r_new, double* diff, int n) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double local = 0.0;
    if (i < n) {
        local = fabs(r_new[i] - r_old[i]);
    }
    sdata[tid] = local;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(diff, sdata[0]);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " MATRIX.mtx\n";
        return EXIT_FAILURE;
    }

    char* filepath = argv[1];
    CSRMatrix<double, int32_t>csr = read_mtx(filepath);
    const int n = csr.nrows;
    const double alpha = 0.85;
    const double tolerance = 1e-6;
    const int max_iter = 1000;
    const int block_size = 256;

    // Host CSR of A (original graph)
    std::vector<int> row_ptr(csr.row_ptr, csr.row_ptr+csr.nrows+1);
    std::vector<int> col_idx(csr.col_idx, csr.col_idx+csr.nnz);
    std::vector<double> values(row_ptr[n], 1.0);

    // Compute out-degree
    std::vector<int> out_degree(n, 0);
    for (int i = 0; i < n; ++i) {
        out_degree[i] = row_ptr[i+1] - row_ptr[i];
        if (out_degree[i] == 0) out_degree[i] = 1;
    }

    // Build A^T in CSR (transpose)
    std::vector<std::vector<int>> adj_t(n);
    for (int u = 0; u < n; ++u) {
        for (int j = row_ptr[u]; j < row_ptr[u+1]; ++j) {
            adj_t[col_idx[j]].push_back(u);
        }
    }
    std::vector<int> at_row_ptr(n+1, 0);
    std::vector<int> at_col_idx;
    for (int i = 0; i < n; ++i) {
        at_col_idx.insert(at_col_idx.end(), adj_t[i].begin(), adj_t[i].end());
        at_row_ptr[i+1] = at_col_idx.size();
    }
    std::vector<double> at_values(at_col_idx.size(), 1.0);
    int nnz_t = at_col_idx.size();

    // Allocate GPU memory
    double *d_r, *d_r_new, *d_out_degree_inv, *d_temp_y;
    int *d_at_row_ptr, *d_at_col_idx;
    double *d_at_values;

    CUDA_CHECK(cudaMalloc(&d_r, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r_new, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out_degree_inv, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_temp_y, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_at_row_ptr, (n+1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_at_col_idx, nnz_t * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_at_values, nnz_t * sizeof(double)));
    CSRMatrix<double, int32_t> csrMat;

    std::vector<double> h_r(n, 1.0 / n);
    std::vector<double> h_out_degree_inv(n);
    for (int i = 0; i < n; ++i) {
        h_out_degree_inv[i] = 1.0 / out_degree[i];
    }
    CUDA_CHECK(cudaMemcpy(d_r, h_r.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_degree_inv, h_out_degree_inv.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_at_row_ptr, at_row_ptr.data(), (n+1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_at_col_idx, at_col_idx.data(), nnz_t * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_at_values, at_values.data(), nnz_t * sizeof(double), cudaMemcpyHostToDevice));
    csrMat.col_idx = d_at_col_idx;
    csrMat.row_ptr = d_at_row_ptr;
    csrMat.val = d_at_values;
    csrMat.nrows = csr.ncols;
    csrMat.ncols = csr.nrows;
    csrMat.nnz = csr.nnz;
    // Initialize r = 1/n

    auto start = std::chrono::high_resolution_clock::now();
    // cuSPARSE setup
    AtBsr<double, int32_t>atbsr;
    CSRToAtBSRProduct(csrMat, atbsr, -1);
    // cusparseDnVecDescr_t vecX, vecY;
    // CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, n, d_temp_y, CUDA_R_64F));
    // CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, n, d_r_new, CUDA_R_64F));

    size_t bufferSize = 0;
    double spmv_alpha = 1.0, spmv_beta = 0.0;


    double *d_r_curr = d_r;
    double *d_r_next = d_r_new;
    double *d_diff = nullptr;
    CUDA_CHECK(cudaMalloc(&d_diff, sizeof(double)));

    double inv_n = (1.0 - alpha) / n;
    int grid_size = (n + block_size - 1) / block_size;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "setup time: " << duration.count() << " us\n";
    for (int iter = 0; iter < max_iter; ++iter) {
        // y = D^{-1} * r_curr
        compute_y<<<grid_size, block_size>>>(d_r_curr, d_out_degree_inv, d_temp_y, n);

        // Update cuSPARSE vector pointers
        // CUSPARSE_CHECK(cusparseDnVecSetValues(vecX, d_temp_y));
        // CUSPARSE_CHECK(cusparseDnVecSetValues(vecY, d_r_next));
        // auto startspmv = std::chrono::high_resolution_clock::now();
        // SpMV: r_next = A^T * y
        // CUSPARSE_CHECK(cusparseSpMV(
        //     handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        //     &spmv_alpha, matA, vecX, &spmv_beta, vecY,
        //     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer
        // ));
        AtSpmvProduct(atbsr, d_temp_y, d_r_next, alpha, spmv_beta);
        // cudaDeviceSynchronize();
        // auto endspmv = std::chrono::high_resolution_clock::now();
        // auto durationspmv = std::chrono::duration_cast<std::chrono::microseconds>(endspmv - startspmv);
        // printf("spmv time: %d us\n", durationspmv.count());
        // r_next = alpha * r_next + (1 - alpha)/n
        pagerank_update<<<grid_size, block_size>>>(d_r_next, alpha, inv_n, n);

        // // Optional: convergence check
        // /*
        if (iter % 10 == 0) {
            CUDA_CHECK(cudaMemset(d_diff, 0, sizeof(double)));
            compute_diff<<<grid_size, block_size, block_size * sizeof(double)>>>(
                d_r_curr, d_r_next, d_diff, n
            );
            double h_diff;
            CUDA_CHECK(cudaMemcpy(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost));
            if (h_diff < tolerance) {
                std::cout << "Converged at iter " << iter + 1 << ", diff = " << h_diff << std::endl;
                break;
            }
        // */
        }
        std::swap(d_r_curr, d_r_next);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - end);
    std::cout << "iter time: " << duration1.count() << " us\n";
    // Copy result
    std::vector<double> result(n);
    CUDA_CHECK(cudaMemcpy(result.data(), d_r_curr, n * sizeof(double), cudaMemcpyDeviceToHost));

    std::cout << "PageRank scores:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "Node " << i << ": " << result[i] << "\n";
    }

    // Cleanup
    cudaFree(d_r); cudaFree(d_r_new); cudaFree(d_temp_y);
    cudaFree(d_out_degree_inv); cudaFree(d_at_row_ptr);
    cudaFree(d_at_col_idx); cudaFree(d_at_values);
    // cudaFree(d_buffer); 
    cudaFree(d_diff);
    // cusparseDestroySpMat(matA);
    // cusparseDestroyDnVec(vecX); cusparseDestroyDnVec(vecY);
    // cusparseDestroy(handle);

    return 0;
}