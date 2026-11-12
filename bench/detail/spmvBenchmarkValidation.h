#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace atspmv::bench {
// Compute both CPU references once, then reuse the same row-scaled FP64
// tolerance for direct execution and CUDA Graph replay of every candidate.
class SpmvReference {
public:
    SpmvReference(const std::vector<int>& ptr, const std::vector<int>& columns,
                  const std::vector<double>& values) : ptr_(ptr) {
        for (int input = 0; input < 2; ++input) {
            sums_[input].resize(ptr.size() - 1);
            magnitudes_[input].resize(ptr.size() - 1);
            for (size_t row = 0; row + 1 < ptr.size(); ++row) {
                for (int j = ptr[row]; j < ptr[row + 1]; ++j) {
                    long double term = static_cast<long double>(values[j]) * inputValue(columns[j], input);
                    sums_[input][row] += term;
                    magnitudes_[input][row] += std::abs(term);
                }
            }
        }
    }

    static double inputValue(int column, int input) {
        return ((column * (input ? 31LL : 23LL) + 7) % 113 - 56) / 67.;
    }

    void verify(const std::vector<double>& result, int input, const std::string& label) const {
        for (size_t row = 0; row < result.size(); ++row) {
            long double tolerance = 8 * std::numeric_limits<double>::epsilon() *
                (ptr_[row + 1] - ptr_[row] + 32.L) * magnitudes_[input][row] + 1e-12L;
            if (!std::isfinite(result[row]) || std::abs(result[row] - sums_[input][row]) > tolerance)
                throw std::runtime_error("incorrect " + label + " row=" + std::to_string(row));
        }
    }

private:
    const std::vector<int>& ptr_;
    std::vector<long double> sums_[2], magnitudes_[2];
};
} // namespace atspmv::bench
