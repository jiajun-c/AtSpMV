#pragma once
#include "sptrsvSelection.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>

namespace atspmv {
// Experimental FP64 exact lower-triangular CSR with int32 indices. Each row
// must have sorted unique columns and a finite nonzero explicit diagonal.
struct SptrsvHostCsr {
    int rows = 0;
    std::vector<int> row_ptr, col_idx;
    std::vector<double> values;
};
struct SptrsvPlanOptions {
    bool enable_cached_tile_model = false; // experimental H100 PCIe solve-only heuristic
};
struct SptrsvPlanInfo {
    SptrsvBackend backend = SptrsvBackend::CuSparse;
    int rows = 0, nnz = 0, levels = -1, device = -1; // -1: analysis skipped
    std::size_t cached_bytes = 0;
};
struct SptrsvPlanState;
// Binds device-resident RHS and output pointers for its lifetime. Their values
// may change between executions. Pointers must not overlap. Create outside
// CUDA Graph capture; keep the plan alive until all queued work has finished.
class SptrsvPlan {
public:
    SptrsvPlan();
    ~SptrsvPlan();
    SptrsvPlan(SptrsvPlan&&) noexcept;
    SptrsvPlan& operator=(SptrsvPlan&&) noexcept;
    SptrsvPlan(const SptrsvPlan&) = delete;
    SptrsvPlan& operator=(const SptrsvPlan&) = delete;
    const SptrsvPlanInfo& info() const;
private:
    std::unique_ptr<SptrsvPlanState> state_;
    friend SptrsvPlan makeSptrsvPlan(const SptrsvHostCsr&,const double*,double*,const SptrsvPlanOptions&,cudaStream_t);
    friend void executeSptrsv(const SptrsvPlan&,cudaStream_t);
};
// Defaults to cuSPARSE. The opt-in experimental H100 PCIe solve-only model
// falls back for unsupported cases. Construction may synchronize the stream.
SptrsvPlan makeSptrsvPlan(const SptrsvHostCsr&,const double* rhs,double* out,
                          const SptrsvPlanOptions& options={},cudaStream_t stream=nullptr);
// One plan per stream; matrix, pointers and stream must remain valid. No
// allocation or explicit synchronization here; supports CUDA Graph capture.
void executeSptrsv(const SptrsvPlan&,cudaStream_t stream=nullptr);
} // namespace atspmv
