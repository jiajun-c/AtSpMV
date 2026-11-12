#include "atspmv.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

static void run(const std::vector<int>& lengths, int cols, int forced_tile) {
    std::vector<int> offsets(1, 0), indices;
    std::vector<double> values, x(cols), result(lengths.size(), 9.0), reference(lengths.size(), 0.0);
    for (int i = 0; i < cols; ++i) x[i] = 0.25 + (i % 11) * 0.125;
    for (std::size_t row = 0; row < lengths.size(); ++row) {
        for (int j = 0; j < lengths[row]; ++j) {
            const int col = (row * 19 + j * 7) % cols;
            const double value = (j % 9 - 4) * 0.25;
            indices.push_back(col); values.push_back(value);
            reference[row] += value * x[col];
        }
        offsets.push_back(values.size());
    }
    CSRMatrix<double, int32_t> csr;
    csr.nrows = lengths.size(); csr.ncols = cols; csr.nnz = values.size();
    csr.row_ptr = offsets.data(); csr.col_idx = indices.data(); csr.val = values.data();
    double time = -1, pre = -1;
    int64_t bytes = -1;
    Spmv(csr, x.data(), result.data(), 2, 3, time, pre, bytes, forced_tile);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    for (std::size_t row = 0; row < lengths.size(); ++row)
        if (!std::isfinite(result[row]) || std::abs(result[row] - reference[row]) >
            1.e-9 * (1 + std::abs(reference[row])))
            throw std::runtime_error("Spmv mismatch at row " + std::to_string(row) +
                                     ", tile=" + std::to_string(forced_tile));
}
int main() {
    try {
        run({0, 0, 0}, 32, -1);
        run(std::vector<int>(128, 4), 100, -1);
        run(std::vector<int>(128, 4), 40000, -1);
        run({0, 2, 0, 3, 4, 0, 5}, 100, -1);
        for (int cols : {100, 40000})
            for (int tile = -1; tile < 6; ++tile)
                run({0, 1, 17, 0, 33, 3, 8192}, cols, tile);
        std::cout << "Spmv host API regression passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
