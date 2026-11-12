#include "absoluteReferenceModel.h"
#include <array>
#include <iomanip>
#include <iostream>

int main() {
    std::array<double, 38> features{};
    std::cout << std::setprecision(17);
    while (std::cin >> features[0]) {
        for (std::size_t i = 1; i < features.size(); ++i)
            if (!(std::cin >> features[i])) return 2;
        std::cout << atspmv::predictAbsoluteReference(features) << '\n';
    }
}
