#include "detail/planModelSelection.h"
#include "unifiedSpmvModel.h"
#include "vectorModelCalibration.h"
#include "vectorModelCalibrationV2.h"
#include "absoluteReferenceModel.h"
#include "mediumRowModel.h"
#include "detail/vectorModelFeaturesV2.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace atspmv::detail {
namespace {
struct ModelCandidate {
    SpmvBackend backend;
    int tile, balanced, feature_tile;
    LongRowConfig long_rows;
};

// The order must match the model table in hybridModelCalibration.h.
constexpr ModelCandidate candidates[] = {
    {SpmvBackend::AtBsr, 3, 0, 3, {8,128,1,1024}},
    {SpmvBackend::AtBsr, 0, 1, 0, {8,128,1,1024}},
    {SpmvBackend::CsrDsm, -1, -1, 3, {8,256,1,1024}},
    {SpmvBackend::CsrTwoStage, -1, -1, 3, {8,256,1,1024}},
    {SpmvBackend::CuSparse, -1, -1, 3, {8,128,1,1024}},
    {SpmvBackend::HybridDsmAtBsr, 3, 0, 3, {8,256,1,1024}},
    {SpmvBackend::HybridDsmAtBsr, 5, 0, 5, {8,256,1,1024}},
    {SpmvBackend::HybridDsmAtBsr, 5, 0, 5, {8,128,1,4096}},
};
}

PlanModelChoice choosePlanModel(const std::vector<int>& row_ptr, int cols,
                                double reference_us, const SpmvPlanOptions& options) {
    const auto f1024 = collectHybridModelFeatures(row_ptr, cols, 1024);
    const auto f4096 = collectHybridModelFeatures(row_ptr, cols, 4096);
    PlanModelChoice choice;
    auto features_v2 = (options.vector_model_v2 || options.vector_model_graph || options.absolute_reference_model)
        ? vectorModelFeaturesV2(row_ptr, f1024, f4096) : std::array<double, 39>{};
    if (options.absolute_reference_model) {
        std::array<double, 38> reference_features{};
        std::copy_n(features_v2.begin(), reference_features.size(), reference_features.begin());
        reference_us = predictAbsoluteReference(reference_features);
        choice.absolute_reference_model_used = true;
    }
    choice.reference_us = reference_us;
    std::array<double, 8> ratios{};
    int best = -1;
    double best_ratio = 1.0;

    for (int i = 0; i < 8; ++i) {
        const auto& features = i == 7 ? f4096 : f1024;
        if (i >= 5 && !unifiedHybridEligible(features)) continue;
        const auto& candidate = candidates[i];
        ratios[i] = predictUnifiedRatio(i, unifiedModelInputs(features, candidate.feature_tile));
        choice.predictions.push_back({candidate.backend, candidate.tile, candidate.balanced,
                                      candidate.long_rows, ratios[i], ratios[i] * reference_us});
        if (i < 5 && ratios[i] < 0.95 && ratios[i] < best_ratio) {
            best = i;
            best_ratio = ratios[i];
        }
    }

    const double hybrid_gate = 1.0 - std::max(0.10, options.hybrid_min_time_reduction);
    int best_hybrid = -1;
    double hybrid_ratio = best_ratio * hybrid_gate;
    for (int i = 5; i < 8; ++i) {
        const auto& features = i == 7 ? f4096 : f1024;
        if (unifiedHybridEligible(features) && ratios[i] < hybrid_ratio) {
            best_hybrid = i;
            hybrid_ratio = ratios[i];
        }
    }
    if (best_hybrid >= 0) best = best_hybrid;

    double selected_ratio = best_hybrid >= 0 ? hybrid_ratio : best_ratio;
    if (options.vector_model_select) {
        const auto shared = unifiedModelInputs(f1024, 3);
        std::array<double, 15> features{};
        std::copy(shared.begin(), shared.end(), features.begin());
        double vector_best = (best_hybrid >= 0 ? hybrid_ratio : best_ratio) *
                             (1.0 - options.vector_model_min_time_reduction);
        for (int width : {1, 2, 4, 8, 16, 32}) {
            features[14] = std::log2(double(width));
            features_v2[38] = features[14];
            const double ratio = options.vector_model_graph ? predictVectorRatioGraph(features_v2)
                               : options.vector_model_v2 ? predictVectorRatioV2(features_v2)
                                                         : predictVectorRatio(features);
            choice.predictions.push_back({SpmvBackend::CsrVector, -1, -1, {}, ratio,
                                          ratio * reference_us, width});
            if (ratio < vector_best) {
                vector_best = ratio;
                choice.vector_width = width;
                selected_ratio = ratio;
            }
        }
        choice.vector_model_used = true;
        choice.vector_model_v2_used = options.vector_model_v2;
        choice.vector_model_graph_used = options.vector_model_graph;
    }

    if (options.medium_row_model) {
        const auto f128 = collectHybridModelFeatures(row_ptr, cols, 128);
        choice.medium_row_model_used = true;
        if (f128.long_rows > f1024.long_rows) {
            double threshold = selected_ratio * hybrid_gate;
            for (int tile = 0; tile < 6; ++tile) {
                const auto x128 = unifiedModelInputs(f128, tile);
                const auto x1024 = unifiedModelInputs(f1024, tile);
                for (int parts : {4, 8}) {
                    std::array<double, 30> features{};
                    std::copy(x128.begin(), x128.end(), features.begin());
                    std::copy(x1024.begin(), x1024.end(), features.begin() + 14);
                    features[28] = tile;
                    features[29] = std::log2(double(parts));
                    const double ratio = predictMediumRowRatio(features);
                    const LongRowConfig config{parts, 256, 4, 128};
                    choice.predictions.push_back({SpmvBackend::HybridDsmAtBsr, tile, 0,
                                                  config, ratio, ratio * reference_us});
                    if (ratio < threshold) {
                        threshold = ratio;
                        choice.selected = true;
                        choice.backend = SpmvBackend::HybridDsmAtBsr;
                        choice.tile_index = tile;
                        choice.balanced = 0;
                        choice.long_rows = config;
                        choice.vector_width = 0;
                    }
                }
            }
            if (choice.selected) return choice;
        }
    }

    if (choice.vector_width) {
        choice.selected = true;
        choice.backend = SpmvBackend::CsrVector;
    } else if (best >= 0) {
        const auto& candidate = candidates[best];
        choice.selected = true;
        choice.backend = candidate.backend;
        choice.tile_index = candidate.tile;
        choice.balanced = candidate.balanced;
        choice.long_rows = candidate.long_rows;
    }
    return choice;
}

} // namespace atspmv::detail
