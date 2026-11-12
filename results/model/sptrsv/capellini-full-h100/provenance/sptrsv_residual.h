#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

struct SptrsvResidual {
    long double normwise = 0;
    long double componentwise = 0;
};

// Evaluate residuals with wider host arithmetic. Componentwise scaling makes
// small rows visible even when a different row dominates the global norm.
inline SptrsvResidual sptrsv_residual(const std::vector<int>& ptr,
    const std::vector<int>& col, const std::vector<double>& val,
    const std::vector<double>& rhs, const std::vector<double>& x) {
    long double rn=0, an=0, xn=0, bn=0, componentwise=0;
    for(std::size_t i=0;i<rhs.size();++i) {
        long double ax=0, row_norm=0, denominator=std::abs((long double)rhs[i]);
        for(int p=ptr[i];p<ptr[i+1];++p) {
            long double a=val[p], v=x[col[p]];
            ax+=a*v; row_norm+=std::abs(a); denominator+=std::abs(a)*std::abs(v);
        }
        long double residual=std::abs(ax-(long double)rhs[i]);
        long double relative=denominator>0 ? residual/denominator :
            (residual==0 ? 0 : std::numeric_limits<long double>::infinity());
        componentwise=std::max(componentwise,relative);
        rn=std::max(rn,residual); an=std::max(an,row_norm);
        xn=std::max(xn,std::abs((long double)x[i])); bn=std::max(bn,std::abs((long double)rhs[i]));
    }
    long double denominator=an*xn+bn;
    return {denominator>0 ? rn/denominator : (rn==0 ? 0 : std::numeric_limits<long double>::infinity()), componentwise};
}
