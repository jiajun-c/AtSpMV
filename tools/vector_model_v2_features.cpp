#include "detail/vectorModelFeaturesV2.h"
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    int rows, cols;
    if (!(std::cin >> rows >> cols) || rows < 0 || cols < 0) return 2;
    std::vector<int> offsets(rows + 1, 0);
    for (int i = 0, count; i < rows; ++i) {
        if (!(std::cin >> count) || count < 0) return 2;
        offsets[i + 1] = offsets[i] + count;
    }
    auto f1024 = atspmv::collectHybridModelFeatures(offsets, cols, 1024);
    auto f4096 = atspmv::collectHybridModelFeatures(offsets, cols, 4096);
    auto features = atspmv::detail::vectorModelFeaturesV2(offsets, f1024, f4096);
    std::cout << std::setprecision(17);
    for (int i = 0; i < 38; ++i) std::cout << features[i] << (i == 37 ? '\n' : ' ');
}
