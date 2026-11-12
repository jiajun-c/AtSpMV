// CPU-only model decision timing; no CUDA context, matrix packing or SpMV.
#include "detail/planModelSelection.h"
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

int main() {
    std::mt19937 rng(87264);
    volatile double sink = 0;
    std::cout << "rows,mode,iterations,mean_us\n";
    for (int rows : {100, 10000, 300000}) {
        std::vector<int> ptr(1, 0);
        for (int i = 0; i < rows; ++i) {
            // Keep nnz within the current H100 model scope (<= 1 million),
            // while retaining a few medium rows at every size.
            int length = rows == 100 ? int(rng()%500) : rows == 10000 ? int(rng()%100) : int(rng()%6);
            if (rows >= 10000 && i%1024 == 0) length += 128 + int(rng()%128);
            ptr.push_back(ptr.back() + length);
        }
        for (int mode = 0; mode < 3; ++mode) {
            atspmv::SpmvPlanOptions options;
            options.vector_model_select = mode > 0;
            options.vector_model_v2 = mode > 0;
            options.medium_row_model = mode == 2;
            const int iterations = rows == 300000 ? 10 : rows == 10000 ? 100 : 1000;
            const auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < iterations; ++i) {
                const auto choice = atspmv::detail::choosePlanModel(ptr, 12345, 1., options);
                sink += choice.predictions.front().time_ratio_to_default;
            }
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-begin).count()/iterations;
            std::cout << rows << ',' << (mode == 0 ? "base" : mode == 1 ? "vector_v2" : "medium")
                      << ',' << iterations << ',' << us << '\n';
        }
    }
    return sink > 0 ? 0 : 1;
}
