#pragma once
#include <array>

namespace atspmv {
// H100 PCIe warm-Graph relative CSR vector time model. Inputs are the same
// 14 matrix features as unifiedModelInputs(f1024, 3), followed by log2(width).
double predictVectorRatio(const std::array<double,15>& features);
}
