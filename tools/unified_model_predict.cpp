#include "unifiedSpmvModel.h"
#include <iomanip>
#include <iostream>
int main() {
    int candidate;
    while(std::cin>>candidate) {
        std::array<double,14> x;
        for(double& v:x)if(!(std::cin>>v))return 2;
        std::cout<<std::setprecision(17)<<atspmv::predictUnifiedRatio(candidate,x)<<'\n';
    }
    return std::cin.eof()?0:2;
}
