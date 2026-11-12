#pragma once

#include <cstddef>

namespace atspmv {
// Experimental solve-only guard, calibrated on a small H100 PCIe pilot.
// This does not include matrix packing, graph setup, or expected reuse count.
enum class SptrsvBackend { CuSparse, CachedTile };
struct SptrsvSelectionFeatures {
    int rows = 0;
    int levels = 0;
    std::size_t cached_bytes = 0;
    std::size_t shared_limit_bytes = 0;
    bool h100_pcie = false;
};
inline SptrsvBackend selectSptrsvBackend(const SptrsvSelectionFeatures& f) {
    if (!f.h100_pcie || f.rows <= 0 || f.rows > 4000 || f.levels < 8 ||
        f.levels > f.rows || f.cached_bytes == 0 || f.cached_bytes > f.shared_limit_bytes ||
        static_cast<double>(f.rows) / f.levels > 256.)
        return SptrsvBackend::CuSparse;
    return SptrsvBackend::CachedTile;
}
// A wider dependency level exposes enough independent row groups to offset
// the extra block-wide synchronization cost of a larger CTA on H100 PCIe.
inline int selectSptrsvTileThreads(int rows, int levels) {
    return rows > 0 && levels > 0 && rows / levels >= 32 ? 1024 : 512;
}
} // namespace atspmv
