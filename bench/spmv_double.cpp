#include "amgT.h"
#include "atspmv.h"
#include "csrReader.h"
#include "cusp.h"
#include "dasp.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " MATRIX.mtx\n";
        return EXIT_FAILURE;
    }

    const char* filepath = argv[1];
    CSRMatrix<double, int32_t> csr = read_mtx(filepath);
    double cuspTime, cuspPreTime, atTime, atPreTime;
    const std::string matrix_name = std::filesystem::path(filepath).filename().string();

    constexpr int warmup = 0;
    constexpr int repeat = 1;
    std::vector<double> input(csr.ncols, 1.0);
    std::vector<double> reference(csr.nrows, 0.0);
    double* x = input.data();
    double* goldenY = reference.data();
    int64_t atBuffSize = 0, cuspBuffSize = 0, amgTBuffSize;

    cusparse_spmvfp64(csr, x, goldenY, repeat, cuspTime, cuspPreTime, cuspBuffSize);

    std::vector<double> output(csr.nrows, 0.0);
    double* aty = output.data();
    Spmv(csr, x, aty, warmup, repeat, atTime, atPreTime, atBuffSize);
    double dasp_avg_time, dasp_pre_time;
    std::vector<int> new_order(csr.nrows);
    char dasp_filename[] = "";
    spmv_all(dasp_filename, csr.val, csr.row_ptr, csr.col_idx, x, aty, new_order.data(), csr.nrows, csr.ncols, csr.nnz, 4, 0.75, 256, dasp_avg_time,dasp_pre_time);

    double amgTtime, amgTPretime;
    amgT_spmv_fp64(csr.row_ptr, csr.col_idx, csr.val, x, aty, csr.nrows, csr.ncols, csr.nnz, repeat, amgTtime, amgTPretime, amgTBuffSize);
    std::cout << matrix_name <<  "," << csr.nnz << "," << atBuffSize << "," <<  cuspBuffSize << "," << amgTBuffSize << "\n";
    printf("stand : %f cuspPre:%f atPreTime:%f amgTPretime:%f\n",csr.stand, cuspPreTime, atPreTime,amgTPretime);
    printf("time: cusp:%f,at:%f,amg:%f dasp: %f\n",cuspTime, atTime, amgTtime, dasp_avg_time);
    delete[] csr.val;
    delete[] csr.row_ptr;
    delete[] csr.col_idx;
    return EXIT_SUCCESS;
}
