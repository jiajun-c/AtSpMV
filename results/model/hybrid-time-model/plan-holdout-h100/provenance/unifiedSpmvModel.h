#pragma once

#include "hybridModelFeatures.h"
#include "hybridModelCalibration.h"
#include <array>
#include <cmath>
#include <stdexcept>

namespace atspmv {
// Predict warm Graph time / default AtBSR time, using the fixed candidate
// order documented in fit_unified_selector.py. This is a relative model; it
// is not an absolute-time estimate and is not portable across GPU profiles.
inline std::array<double,14> unifiedModelInputs(const HybridModelFeatures& f, int tile) {
    if(tile<0||tile>5)throw std::invalid_argument("invalid model tile");
    const double raw[]={double(f.rows),double(f.cols),double(f.nnz),double(f.long_rows),
        double(f.long_nnz),double(f.longest),double(f.longest_short),f.long_length_squared,
        double(f.short_tiles[tile]),double(f.short_max_tiles[tile])};
    std::array<double,14> x{};
    for(int i=0;i<10;++i)x[i]=std::log1p(raw[i]);
    x[10]=double(f.long_rows)/std::max(f.rows,1);
    x[11]=double(f.long_nnz)/std::max<std::int64_t>(f.nnz,1);
    x[12]=double(f.short_tiles[tile])*32/std::max<std::int64_t>(f.nnz,1);
    x[13]=f.cols<32767?1.:0.;
    return x;
}

inline double predictUnifiedRatio(int candidate,const std::array<double,14>& x) {
    using namespace hybrid_model_data;
    if(candidate<0||candidate>=int(sizeof(models)/sizeof(models[0])))throw std::invalid_argument("invalid model candidate");
    const auto& model=models[candidate];double result=model.baseline;
    for(std::size_t t=0;t<model.count;++t) {
        const Node* nodes=model.trees[t].nodes;int node=0;
        while(!nodes[node].leaf) {
            const auto& current=nodes[node];
            node=x[current.feature]<=current.threshold?current.left:current.right;
        }
        result+=nodes[node].value;
    }
    return std::exp(result);
}
} // namespace atspmv
