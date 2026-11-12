#include "mediumRowModel.h"
#include <array>
#include <iomanip>
#include <iostream>

int main() {
    std::array<double, 30> features{};
    while (std::cin >> features[0]) {
        for (std::size_t i = 1; i < features.size(); ++i)
            if (!(std::cin >> features[i])) return 1;
        std::cout << std::setprecision(17) << atspmv::predictMediumRowRatio(features) << '\n';
    }
    return std::cin.eof() ? 0 : 1;
}
