#include "detail/planModelSelection.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    std::vector<int> rows(101,0);
    for (int i=0;i<100;++i) rows[i+1]=rows[i]+(i==5?300:3+i%12);
    atspmv::SpmvPlanOptions options;
    options.vector_model_select=true;
    options.vector_model_v2=true;
    auto baseline=atspmv::detail::choosePlanModel(rows,2048,4.0,options);
    options.absolute_reference_model=true;
    auto learned=atspmv::detail::choosePlanModel(rows,2048,4.0,options);
    if (!learned.absolute_reference_model_used || !std::isfinite(learned.reference_us) || learned.reference_us<=0)
        throw std::runtime_error("absolute reference was not used");
    if (baseline.selected!=learned.selected || baseline.backend!=learned.backend ||
        baseline.tile_index!=learned.tile_index || baseline.balanced!=learned.balanced ||
        baseline.vector_width!=learned.vector_width || baseline.predictions.size()!=learned.predictions.size() ||
        baseline.long_rows.parts!=learned.long_rows.parts || baseline.long_rows.threads!=learned.long_rows.threads ||
        baseline.long_rows.rows_per_group!=learned.long_rows.rows_per_group || baseline.long_rows.threshold!=learned.long_rows.threshold)
        throw std::runtime_error("reference changed selection");
    for (std::size_t i=0;i<baseline.predictions.size();++i) {
        const auto& old=baseline.predictions[i];const auto& current=learned.predictions[i];
        if (old.time_ratio_to_default!=current.time_ratio_to_default ||
            std::abs(current.estimated_us-current.time_ratio_to_default*learned.reference_us)>1e-12)
            throw std::runtime_error("prediction scaling mismatch");
    }
    std::cout << "absolute reference selection passed\n";
}
