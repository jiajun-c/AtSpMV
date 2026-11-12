#include "hybridModelFeatures.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

// Reference implementation: sort short rows then inspect each group maximum.
static atspmv::HybridModelFeatures reference(const std::vector<int>& offsets, int cols, int threshold) {
    atspmv::HybridModelFeatures f;
    f.rows = int(offsets.size()) - 1; f.cols = cols; f.threshold = threshold; f.nnz = offsets.back();
    std::vector<int> lengths;
    lengths.reserve(f.rows);
    for (int i = 0; i < f.rows; ++i) {
        const int length = offsets[i+1] - offsets[i];
        f.longest = std::max(f.longest, length);
        if (length >= threshold) {
            ++f.long_rows; f.long_nnz += length; f.long_length_squared += double(length)*length;
        } else {
            lengths.push_back(length); f.short_nnz += length; f.longest_short = std::max(f.longest_short, length);
        }
    }
    f.short_rows = int(lengths.size());
    std::sort(lengths.begin(), lengths.end());
    for (int tile = 0; tile < 6; ++tile) {
        const int m = 1 << tile, k = 32 / m;
        for (std::size_t first = 0; first < lengths.size(); first += m) {
            const auto last = std::min(first + m, lengths.size()) - 1;
            const int count = (std::int64_t(lengths[last]) + k - 1) / k;
            f.short_tiles[tile] += count;
            f.short_max_tiles[tile] = std::max(f.short_max_tiles[tile], count);
        }
    }
    return f;
}

static void verify(const std::vector<int>& ptr, int threshold) {
    const auto a = reference(ptr, 12345, threshold);
    const auto b = atspmv::collectHybridModelFeatures(ptr, 12345, threshold);
    if (a.rows != b.rows || a.cols != b.cols || a.nnz != b.nnz || a.threshold != b.threshold ||
        a.long_rows != b.long_rows || a.short_rows != b.short_rows || a.long_nnz != b.long_nnz ||
        a.short_nnz != b.short_nnz || a.longest != b.longest || a.longest_short != b.longest_short ||
        a.long_length_squared != b.long_length_squared || a.short_tiles != b.short_tiles ||
        a.short_max_tiles != b.short_max_tiles) {
        std::cerr << "feature mismatch for threshold " << threshold << ", rows " << a.rows << '\n';
        std::abort();
    }
}

static void benchmark() {
    std::mt19937 rng(87264);
    volatile std::uint64_t sink = 0;
    std::cout << "rows,threshold,reference_us,histogram_us,speedup\n";
    for (int rows : {100, 10000, 300000}) {
        std::vector<int> ptr(1, 0);
        for (int i = 0; i < rows; ++i) ptr.push_back(ptr.back() + int(rng()%500));
        const int iterations = rows < 1000 ? 5000 : rows < 100000 ? 200 : 10;
        for (int threshold : {128, 1024, 4096}) {
            auto measure = [&](auto fn) {
                const auto begin = std::chrono::steady_clock::now();
                for (int i = 0; i < iterations; ++i) sink += fn(ptr, 12345, threshold).short_tiles[3];
                return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-begin).count()/iterations;
            };
            const double old_us = measure(reference);
            const double new_us = measure(atspmv::collectHybridModelFeatures);
            std::cout << rows << ',' << threshold << ',' << old_us << ',' << new_us << ',' << old_us/new_us << '\n';
        }
    }
    if (sink == 0) std::abort();
}

int main(int argc, char**) {
    if (argc > 1) { benchmark(); return 0; }
    try {
        atspmv::collectHybridModelFeatures({0, 2, 1}, 1, 128);
        std::abort();
    } catch (const std::invalid_argument&) {}
    std::mt19937 rng(87264);
    for (int n : {0, 1, 2, 7, 31, 32, 33, 63, 64, 65, 127, 1000, 10000}) {
        for (int iteration = 0; iteration < 30; ++iteration) {
            std::vector<int> ptr(1, 0);
            for (int i = 0; i < n; ++i) {
                const int length = iteration % 3 == 0 ? 0 : iteration % 3 == 1 ? rng()%150 : rng()%5000;
                ptr.push_back(ptr.back() + length);
            }
            for (int threshold : {0, 1, 2, 31, 32, 33, 128, 256, 1024, 4096, 65536, 65537})
                verify(ptr, threshold);
        }
    }
    std::cout << "hybrid model features agree with sorted reference\n";
}
