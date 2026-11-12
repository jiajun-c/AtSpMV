#include "vectorModelCalibration.h"
#include <array>
#include <iomanip>
#include <iostream>

int main() {
    std::array<double,15> features{};
    std::cout<<std::setprecision(17);
    while(std::cin>>features[0]) {
        for(int i=1;i<15;++i)if(!(std::cin>>features[i]))return 2;
        std::cout<<atspmv::predictVectorRatio(features)<<'\n';
    }
    return 0;
}
