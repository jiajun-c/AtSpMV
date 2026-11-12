#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
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
    // All short lengths are in [0, threshold). A bounded histogram lets us
    // obtain the sorted group maxima without sorting one item per row.
    // Keep the general path for unusually large or nonpositive thresholds.
    const bool use_histogram = f.rows >= 512 && threshold > 0 && threshold <= 65536 &&
                               std::int64_t(f.rows) * 16 >= threshold;
    std::vector<std::uint64_t> histogram(use_histogram ? threshold : 0, 0);
    std::vector<int> short_lengths;
    if (!use_histogram) short_lengths.reserve(f.rows);
    for (int row = 0; row < f.rows; ++row) {
        const int length = offsets[row+1] - offsets[row];
        if (length < 0) throw std::invalid_argument("nonmonotonic CSR offsets");
        f.longest = std::max(f.longest, length);
        if (length >= threshold) {
            ++f.long_rows;
            f.long_nnz += length;
            f.long_length_squared += double(length)*length;
        } else {
            if (use_histogram) ++histogram[length];
            else short_lengths.push_back(length);
            f.short_nnz += length;
            f.longest_short = std::max(f.longest_short, length);
        }
    }
    f.short_rows = f.rows - f.long_rows;
    if (f.short_rows == 0) return f;
    if (use_histogram) {
        std::uint64_t previous = 0;
        for (int length = 0; length < threshold; ++length) {
            const auto current = previous + histogram[length];
            if (current != previous) {
                for (int tile = 0; tile < 6; ++tile) {
                    const int m = 1 << tile, k = 32 / m;
                    auto groups = current / m - previous / m;
                    // The final incomplete group also uses its final row.
                    if (current == static_cast<std::uint64_t>(f.short_rows) && current % m) ++groups;
                    if (groups) {
                        const int count = (static_cast<std::int64_t>(length) + k - 1) / k;
                        f.short_tiles[tile] += groups * count;
                        f.short_max_tiles[tile] = std::max(f.short_max_tiles[tile], count);
                    }
                }
            }
            previous = current;
        }
        return f;
    }
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
