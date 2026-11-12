#pragma once

#include "hybridModelFeatures.h"
#include "unifiedSpmvModel.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace atspmv::detail {

struct VectorTailFeatures {
    int long_rows = 0;
    std::int64_t long_nnz = 0;
    int longest_short = 0;
    double long_length_squared = 0;
};

// Compute both additional thresholds in one pass, without re-sorting rows.
inline std::array<VectorTailFeatures, 2> vectorTailFeatures(const std::vector<int>& row_ptr) {
    std::array<VectorTailFeatures, 2> result{};
    constexpr int thresholds[] = {128, 256};
    for (size_t row = 0; row + 1 < row_ptr.size(); ++row) {
        const int length = row_ptr[row + 1] - row_ptr[row];
        for (int i = 0; i < 2; ++i) {
            auto& f = result[i];
            if (length >= thresholds[i]) {
                ++f.long_rows;
                f.long_nnz += length;
                f.long_length_squared += double(length) * length;
            } else {
                f.longest_short = std::max(f.longest_short, length);
            }
        }
    }
    return result;
}

// Matches enriched() in train_vector_accuracy.py. Element 38 is set by the
// caller to log2(vector_width), so all widths share the matrix features.
inline std::array<double, 39> vectorModelFeaturesV2(const std::vector<int>& row_ptr,
                                                     const HybridModelFeatures& f1024,
                                                     const HybridModelFeatures& f4096) {
    std::array<double, 39> result{};
    const auto original = unifiedModelInputs(f1024, 3);
    std::copy(original.begin(), original.end(), result.begin());
    int index = 14;
    for (auto value : f1024.short_tiles) result[index++] = std::log1p(double(value));
    for (auto value : f1024.short_max_tiles) result[index++] = std::log1p(double(value));
    const auto extra = vectorTailFeatures(row_ptr);
    const double rows = std::max(f1024.rows, 1);
    const double nnz = std::max<std::int64_t>(f1024.nnz, 1);
    auto append = [&](int long_rows, std::int64_t long_nnz, int longest_short, double squared) {
        result[index++] = long_rows / rows;
        result[index++] = long_nnz / nnz;
        result[index++] = std::log1p(double(longest_short));
        result[index++] = std::log1p(squared);
    };
    for (const auto& f : extra) append(f.long_rows, f.long_nnz, f.longest_short, f.long_length_squared);
    append(f4096.long_rows, f4096.long_nnz, f4096.longest_short, f4096.long_length_squared);
    return result;
}

} // namespace atspmv::detail
