#pragma once
#include <array>

namespace atspmv {
// H100 warm-Graph medium-row hybrid time divided by default AtBSR time.
double predictMediumRowRatio(const std::array<double, 30>& features);
}
