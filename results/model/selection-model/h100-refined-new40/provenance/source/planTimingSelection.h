#pragma once

#include "spmvPlan.h"
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace atspmv::detail {

inline bool needsMeasuredMargin(SpmvBackend backend) {
    return backend == SpmvBackend::HybridDsmAtBsr ||
           backend == SpmvBackend::CsrDsmVector || backend == SpmvBackend::MixedTile;
}

// Auto applies the margin to hybrid and mixed backends, and excludes DSM
// partitions without long rows. Explicit backend tuning simply picks the
// lowest measured median. `initial` preserves the caller's tie behavior.
inline size_t selectMeasuredCandidate(const std::vector<SpmvCandidateTiming>& candidates,
                                      const std::vector<bool>& eligible,
                                      bool automatic, double margin, size_t initial = 0) {
    if (candidates.empty() || candidates.size() != eligible.size() || initial >= candidates.size())
        throw std::invalid_argument("invalid measured candidate list");
    size_t winner = initial;
    for (size_t i = 0; i < candidates.size(); ++i)
        if (candidates[i].median_us < candidates[winner].median_us) winner = i;
    if (!automatic) return winner;

    size_t regular = candidates.size(), hybrid = candidates.size();
    for (size_t i = 0; i < candidates.size(); ++i) {
        const bool guarded = needsMeasuredMargin(candidates[i].backend);
        if (guarded && !eligible[i]) continue;
        size_t& best = guarded ? hybrid : regular;
        if (best == candidates.size() || candidates[i].median_us < candidates[best].median_us) best = i;
    }
    if (regular == candidates.size()) return winner;
    if (hybrid != candidates.size() && candidates[hybrid].median_us <= candidates[regular].median_us * (1 - margin))
        return hybrid;
    return regular;
}

} // namespace atspmv::detail
