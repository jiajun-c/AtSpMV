#pragma once

#include "spmvPlan.h"
#include <vector>

namespace atspmv::detail {

// Describes a predicted candidate; Plan construction and device memory remain
// owned by the caller. If selected is false, keep the default AtBSR candidate.
struct PlanModelChoice {
    bool selected = false;
    bool vector_model_used = false;
    bool vector_model_v2_used = false;
    bool vector_model_graph_used = false;
    bool medium_row_model_used = false;
    bool absolute_reference_model_used = false;
    double reference_us = 0;
    SpmvBackend backend = SpmvBackend::AtBsr;
    int tile_index = -1;
    int balanced = -1;
    LongRowConfig long_rows{};
    int vector_width = 0;
    std::vector<SpmvModelPrediction> predictions;
};

PlanModelChoice choosePlanModel(const std::vector<int>& row_ptr, int cols,
                                double reference_us, const SpmvPlanOptions& options);

} // namespace atspmv::detail
