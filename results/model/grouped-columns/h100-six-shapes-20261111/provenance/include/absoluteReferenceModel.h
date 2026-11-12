#pragma once
#include <array>

namespace atspmv {
// Experimental H100 PCIe warm-Graph default-AtBSR time in microseconds.
// Feature order is the first 38 values of detail::vectorModelFeaturesV2.
double predictAbsoluteReference(const std::array<double, 38>& features);
}
