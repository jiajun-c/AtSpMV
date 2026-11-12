#pragma once

#include "spmvPlan.h"
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace atspmv::bench {
inline bool labelIn(const std::string& label, std::initializer_list<const char*> choices) {
    for (const char* choice : choices)
        if (label == choice) return true;
    return false;
}

// Experiment labels stay outside the public API. Related labels share defaults;
// only the switches distinguishing each experiment are set below.
inline SpmvPlanOptions benchmarkOptions(const std::string& kind, int tile, int balance,
                                       int parts, int threads, int group, int threshold) {
    SpmvPlanOptions options;
    options.tile_index = tile;
    options.balanced = balance;
    options.long_rows = {parts, threads, group, threshold};
    options.share_atbsr_tuning_storage = std::getenv("ATSPMV_BENCH_SHARE_ATBSR_PACKING") != nullptr;
    options.retry_shared_storage_on_oom = std::getenv("ATSPMV_BENCH_RETRY_SHARED_ON_OOM") != nullptr;

    if (labelIn(kind, {"atbsr", "atbsr_control", "atbsr_m8", "atbsr_m1_balanced", "atbsr_m32",
                       "atbsr_historical", "atbsr_tuning", "atbsr_clear", "atbsr_i32"})) {
        options.backend = SpmvBackend::AtBsr;
    } else if (labelIn(kind, {"atbsr_compressed", "atbsr_grouped", "atbsr_grouped_model"})) {
        options.backend = SpmvBackend::AtBsr;
        options.compress_tile_columns = kind == "atbsr_compressed";
        options.group_tile_columns = !options.compress_tile_columns;
    } else if (kind == "mixed_tile") {
        options.backend = SpmvBackend::MixedTile;
        options.mixed_tile_cap = parts;
        options.long_rows = {8, 128, 1, 1024};
    } else if (labelIn(kind, {"csr_dsm", "csr_dsm_fixed"})) {
        options.backend = SpmvBackend::CsrDsm;
    } else if (labelIn(kind, {"csr_two_stage", "csr_two_stage_fixed"})) {
        options.backend = SpmvBackend::CsrTwoStage;
    } else if (kind == "hybrid_dsm_atbsr") {
        options.backend = SpmvBackend::HybridDsmAtBsr;
    } else if (kind.rfind("csr_vec", 0) == 0) {
        options.backend = SpmvBackend::CsrVector;
        options.vector_width = parts;
        options.long_rows = {8, 128, 1, 1024};
    } else if (kind.rfind("csr_dsm_vec", 0) == 0) {
        options.backend = SpmvBackend::CsrDsmVector;
        options.vector_width = std::stoi(kind.substr(11));
    } else if (labelIn(kind, {"auto_tuned", "auto_tuned_shared", "auto_fast_tuned", "auto_tuned_mixed"})) {
        options.tune = true;
        options.fast_tune = kind == "auto_fast_tuned";
        options.enable_mixed_tile = kind == "auto_tuned_mixed";
        if (kind == "auto_tuned_shared") options.share_atbsr_tuning_storage = true;
    } else if (labelIn(kind, {"auto_model", "auto_vector_model", "auto_vector_model_v2",
                              "auto_vector_model_graph", "auto_vector_model_2pct", "auto_medium_model",
                              "auto_model_fallback_tuned", "auto_model_refined", "auto_model_refined_vector",
                              "auto_refined_v2", "auto_refined_v2_grouped"})) {
        options.model_select = true;
        options.vector_model_select = kind != "auto_model";
        options.vector_model_v2 = labelIn(kind, {"auto_vector_model_v2", "auto_medium_model",
                                               "auto_refined_v2", "auto_refined_v2_grouped"});
        options.model_refine = labelIn(kind, {"auto_model_refined", "auto_model_refined_vector",
                                            "auto_refined_v2", "auto_refined_v2_grouped"});
        options.enable_mixed_tile = labelIn(kind, {"auto_model_refined", "auto_refined_v2",
                                                 "auto_refined_v2_grouped"});
        options.medium_row_model = kind == "auto_medium_model";
        options.vector_model_graph = kind == "auto_vector_model_graph";
        options.model_fallback_tune = kind == "auto_model_fallback_tuned";
        options.group_tile_columns = kind == "auto_refined_v2_grouped";
        if (kind == "auto_vector_model_2pct") options.vector_model_min_time_reduction = .02;
    } else {
        throw std::runtime_error("unknown backend: " + kind);
    }
    return options;
}

inline bool logsTuning(const SpmvPlanOptions& options, const SpmvPlanInfo& info) {
    return options.tune || options.model_refine || info.model_fallback_tune_used;
}

inline std::string resolvedBenchmarkConfig(const std::string& kind, const SpmvPlanOptions& options,
                                          const SpmvPlanInfo& info, const std::string& fallback) {
    if (kind == "atbsr_control")
        return "resolved/" + std::to_string(info.tile_index) + "/" + std::to_string(info.balanced);
    if (kind.rfind("auto_", 0) != 0) return fallback;

    if (kind == "auto_model" && !info.model_used)
        throw std::runtime_error("model not available for input");
    if (labelIn(kind, {"auto_vector_model", "auto_vector_model_2pct"}) && !info.vector_model_used)
        throw std::runtime_error("vector model not available for input");
    if (kind == "auto_vector_model_v2" && !info.vector_model_v2_used)
        throw std::runtime_error("vector model v2 not available for input");
    if (kind == "auto_vector_model_graph" && !info.vector_model_graph_used)
        throw std::runtime_error("graph vector model not available for input");
    if (kind == "auto_medium_model" && !info.medium_row_model_used)
        throw std::runtime_error("medium row model not available for input");
    if (kind == "auto_fast_tuned" && !info.fast_tuning_used)
        throw std::runtime_error("fast tuning not active");
    if (kind == "auto_model_fallback_tuned" && !info.model_fallback_tune_used && !info.vector_model_used)
        throw std::runtime_error("neither model nor fallback tuning active");
    if (options.model_refine && !info.model_refined)
        throw std::runtime_error("model refinement not available");

    std::string config = std::string(spmvBackendName(info.backend)) + "/" +
        std::to_string(info.tile_index) + "/" + std::to_string(info.balanced) + "/" +
        std::to_string(info.long_rows.parts) + "/" + std::to_string(info.long_rows.threads) + "/" +
        std::to_string(info.long_rows.rows_per_group) + "/" + std::to_string(info.long_rows.threshold) +
        "/vw=" + std::to_string(info.vector_width);
    if (options.enable_mixed_tile && (options.tune || options.model_refine))
        config += "/mc=" + std::to_string(info.mixed_tile_cap);
    return config;
}
} // namespace atspmv::bench
