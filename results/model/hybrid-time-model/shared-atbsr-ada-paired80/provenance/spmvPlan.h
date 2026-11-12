#pragma once
#include "matrixFormat.h"
#include <memory>
#include <string>
#include <vector>

namespace atspmv {
enum class SpmvBackend { Auto, AtBsr, CsrDsm, CsrTwoStage, HybridDsmAtBsr, CuSparse, CsrVector, CsrDsmVector };
struct LongRowConfig {
    int parts = 8;
    int threads = 128;
    int rows_per_group = 1;
    int threshold = 1024;
};
struct SpmvPlanOptions {
    SpmvBackend backend = SpmvBackend::Auto;
    bool tune = false;
    // Optional bounded Auto tuning shortlist; full tuning remains the default.
    // Explicit tile/mode or custom long-row lists retain full tuning.
    bool fast_tune = false;
    // Experimental: reuse each balanced AtBSR packing for its normal-mode
    // tuning candidate, reducing peak allocation. Requires full tuning.
    bool share_atbsr_tuning_storage = false;
    // On a recognized GPU allocation failure, optionally retry full tuning
    // once with shared AtBSR candidate storage.
    bool retry_shared_storage_on_oom = false;
    // Opt in to the H100 PCIe warm-Graph model when tune=false. Unsupported
    // hardware/workloads fall back to the existing AtBSR model.
    bool model_select = false;
    int tile_index = -1; // -1: existing model; otherwise 0..5
    int balanced = -1;   // -1: existing model; otherwise 0/1; hybrid supports only 0
    LongRowConfig long_rows{}; // used without tuning
    int vector_width = 4; // threads per CSR row: 1, 2, 4, 8, 16 or 32
    // Empty uses a bounded built-in shortlist; explicit lists override it.
    std::vector<LongRowConfig> tuning_long_rows;
    int warmup_batches = 2;
    int measurement_batches = 7;
    int calls_per_batch = 8;
    // Auto uses an AtBSR/DSM or CSR/DSM mixed path only if its measured time
    // is at least this fraction below the best non-mixed candidate; explicit
    // backend tuning is ungated.
    double hybrid_min_time_reduction = 0.05;
    // Experimental H100 CSR-vector predictor within model_select; defaults off.
    // Requires Auto, model_select=true and tune=false.
    bool vector_model_select = false;
    // Minimum predicted reduction versus the best existing model candidate.
    // Default 0 preserves the originally validated vector-model decision.
    double vector_model_min_time_reduction = 0.0;
    // Explicitly tune Auto when the selected model is out of its supported
    // device/environment/matrix scope. May incur substantial setup and memory.
    bool model_fallback_tune = false;
};
struct SpmvCandidateTiming {
    SpmvBackend backend = SpmvBackend::AtBsr;
    int tile_index = -1, balanced = -1;
    LongRowConfig long_rows{};
    int vector_width = 0;
    double median_us = 0;
};
struct SpmvModelPrediction {
    SpmvBackend backend = SpmvBackend::AtBsr;
    int tile_index = -1, balanced = -1;
    LongRowConfig long_rows{};
    double time_ratio_to_default = 0;
    double estimated_us = 0; // relative estimate multiplied by the AtBSR model
    int vector_width = 0;
};
struct SpmvPlanInfo {
    SpmvBackend backend = SpmvBackend::AtBsr;
    int rows = 0, cols = 0, nnz = 0, device = -1;
    int tile_index = -1, balanced = -1;
    LongRowConfig long_rows{};
    bool measured = false, dsm_available = false;
    bool fast_tuning_used = false;
    int vector_width = 0;
    bool model_used = false;
    bool model_reference_calibrated = false;
    double model_reference_us = 0;
    double setup_us = 0;
    std::vector<SpmvCandidateTiming> timings;
    std::vector<SpmvModelPrediction> model_predictions;
    // Unsupported tile layouts omitted during tuning, with the reason.
    std::vector<std::string> skipped_candidates;
    bool vector_model_used = false;
    bool model_fallback_tune_used = false;
    bool oom_retry_shared_storage_used = false;
};
struct SpmvPlanState;
// Move-only owner of matrix copies, row lists, and scratch. Source CSR can be
// released after construction. Keep the plan alive until queued work and all
// captured graphs using it have finished. One plan per concurrent stream.
class SpmvPlan {
public:
    SpmvPlan();
    ~SpmvPlan();
    SpmvPlan(SpmvPlan&&) noexcept;
    SpmvPlan& operator=(SpmvPlan&&) noexcept;
    SpmvPlan(const SpmvPlan&) = delete;
    SpmvPlan& operator=(const SpmvPlan&) = delete;
    const SpmvPlanInfo& info() const;
private:
    std::unique_ptr<SpmvPlanState> state_;
    friend SpmvPlan makeSpmvPlan(const CSRMatrix<double,int32_t>&, const SpmvPlanOptions&, cudaStream_t);
    friend void executeSpmv(const SpmvPlan&, const double*, double*, cudaStream_t);
};
// Input arrays reside on the current CUDA device. Construction synchronizes
// stream before copying; call outside capture. Auto without tune uses AtBSR's
// existing model. Auto+tune compares compiled/supported backends, including
// cuSPARSE CSR_ALG1 and hybrid DSM/normal-AtBSR when enabled, on internal
// nonuniform X and private Y, using warm CUDA Graph GPU timings (not cold-HBM
// or CPU launch latency). Preparation is excluded from candidate timings.
SpmvPlan makeSpmvPlan(const CSRMatrix<double,int32_t>&,
                     const SpmvPlanOptions& options = {}, cudaStream_t stream = nullptr);
// Computes y=A*x (alpha=1,beta=0 only), observes the current x on every launch.
// No allocation/tuning/synchronization here; supports CUDA Graph capture.
// x/y must be on the plan device and have no overlapping storage.
void executeSpmv(const SpmvPlan&, const double* x, double* y, cudaStream_t stream = nullptr);
const char* spmvBackendName(SpmvBackend);
} // namespace atspmv
