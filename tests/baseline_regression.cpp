#include "amgT.h"
#include "csrReader.h"
#include "cusp.h"
#include "dasp.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: baseline_regression MATRIX.mtx");
        auto matrix = read_mtx(argv[1]);
        std::vector<double> x(matrix.ncols), expected(matrix.nrows), output(matrix.nrows);
        for (int col = 0; col < matrix.ncols; ++col) x[col] = ((col * 23LL + 7) % 113 - 56) / 67.;
        for (int row = 0; row < matrix.nrows; ++row) {
            long double sum = 0;
            for (int i = matrix.row_ptr[row]; i < matrix.row_ptr[row + 1]; ++i)
                sum += static_cast<long double>(matrix.val[i]) * x[matrix.col_idx[i]];
            expected[row] = static_cast<double>(sum);
        }
        auto verify = [&](const char* name, const std::vector<int>* order = nullptr, double scale = 1.0) {
            std::vector<bool> seen(matrix.nrows);
            for (int row = 0; row < matrix.nrows; ++row) {
                const int original = order ? (*order)[row] : row;
                if (original < 0 || original >= matrix.nrows || seen[original])
                    throw std::runtime_error(std::string(name) + ": invalid row order");
                seen[original] = true;
                if (!std::isfinite(output[row]) ||
                    std::abs(output[row] - scale * expected[original]) > 1e-10 * (1 + std::abs(scale * expected[original])))
                    throw std::runtime_error(std::string(name) + ": incorrect row " + std::to_string(original) +
                                             ", got " + std::to_string(output[row]) +
                                             ", expected " + std::to_string(scale * expected[original]));
            }
        };
        double avg = 0, prep = 0;
        int64_t bytes = 0;
        cusparse_spmvfp64(matrix, x.data(), output.data(), 1, avg, prep, bytes);
        verify("cuSPARSE");
        std::fill(output.begin(), output.end(), 0.0);
        amgT_spmv_fp64(matrix.row_ptr, matrix.col_idx, matrix.val, x.data(), output.data(),
                       matrix.nrows, matrix.ncols, matrix.nnz, 1, avg, prep, bytes);
        // The legacy wrapper accumulates all 10 warmups and the timed call.
        // This checks unchanged behavior, not a correct y=A*x interface.
        verify("AMG-T legacy accumulation", nullptr, 11.0);
        std::fill(output.begin(), output.end(), 0.0);
        std::vector<int> order(matrix.nrows);
        char filename[] = "";
        spmv_all(filename, matrix.val, matrix.row_ptr, matrix.col_idx, x.data(), output.data(),
                 order.data(), matrix.nrows, matrix.ncols, matrix.nnz, 4, 0.75, 256, avg, prep);
        verify("DASP", &order);
        delete[] matrix.row_ptr; delete[] matrix.col_idx; delete[] matrix.val;
        std::cout << "DASP and cuSPARSE match the CPU reference; AMG-T retains legacy 11*Ax accumulation\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
