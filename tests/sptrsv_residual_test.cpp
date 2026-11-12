#include "../bench/sptrsv_residual.h"
#include <cmath>

int main() {
    // The large first row hides the error in row two from the global norm.
    std::vector<int> ptr{0,1,2}, col{0,1};
    std::vector<double> val{1e20,1.}, rhs{1e20,1.}, x{1.,2.};
    auto r=sptrsv_residual(ptr,col,val,rhs,x);
    if(!(r.normwise<1e-19L && std::abs(r.componentwise-1.L/3)<1e-18L))return 1;
    x[1]=1.; r=sptrsv_residual(ptr,col,val,rhs,x);
    if(r.normwise!=0 || r.componentwise!=0)return 2;
    // Zero RHS and solution produce a zero residual with zero denominator.
    x={0.,0.}; rhs={0.,0.}; r=sptrsv_residual(ptr,col,val,rhs,x);
    if(r.normwise!=0 || r.componentwise!=0)return 3;
}
