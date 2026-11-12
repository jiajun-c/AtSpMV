#pragma once
#include <array>

namespace atspmv {
// Experimental H100 PCIe warm-Graph vector time ratio; feature order is
// defined by detail::vectorModelFeaturesV2.
double predictVectorRatioV2(const std::array<double, 39>& features);
double predictVectorRatioGraph(const std::array<double, 39>& features);
}
