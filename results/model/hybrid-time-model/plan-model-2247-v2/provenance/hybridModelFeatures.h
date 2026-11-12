#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace atspmv {
// Host-side statistics shared by model fitting and runtime selection. Values
// count stored CSR entries, including duplicates and explicit zeros.
struct HybridModelFeatures {
    int rows = 0, cols = 0, threshold = 0;
    std::int64_t nnz = 0, long_nnz = 0, short_nnz = 0;
    int long_rows = 0, short_rows = 0, longest = 0, longest_short = 0;
    double long_length_squared = 0;
    std::array<std::uint64_t, 6> short_tiles{};
    std::array<int, 6> short_max_tiles{};
};

inline HybridModelFeatures collectHybridModelFeatures(const std::vector<int>& offsets, int cols, int threshold) {
    HybridModelFeatures f;
    f.rows = static_cast<int>(offsets.size()) - 1;
    f.cols = cols;
    f.threshold = threshold;
    f.nnz = offsets.back();
    std::vector<int> short_lengths;
    short_lengths.reserve(f.rows);
    for (int row = 0; row < f.rows; ++row) {
        const int length = offsets[row+1] - offsets[row];
        f.longest = std::max(f.longest, length);
        if (length >= threshold) {
            ++f.long_rows;
            f.long_nnz += length;
            f.long_length_squared += double(length)*length;
        } else {
            short_lengths.push_back(length);
            f.short_nnz += length;
            f.longest_short = std::max(f.longest_short, length);
        }
    }
    f.short_rows = static_cast<int>(short_lengths.size());
    std::sort(short_lengths.begin(), short_lengths.end());
    for (int tile=0; tile<6; ++tile) {
        const int m = 1 << tile, k = 32 / m;
        for (std::size_t first = 0; first < short_lengths.size(); first += m) {
            const std::size_t last = std::min(first + m, short_lengths.size()) - 1;
            const int count = (static_cast<std::int64_t>(short_lengths[last]) + k - 1) / k;
            f.short_tiles[tile] += count;
            f.short_max_tiles[tile] = std::max(f.short_max_tiles[tile], count);
        }
    }
    return f;
}
} // namespace atspmv
