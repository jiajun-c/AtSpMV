#pragma once

#include "atspmvprod.h"

namespace atspmv {

// Experimental Hopper path. No time-model selection or implicit fallback.
// Global is a control with the same cluster/grid-stride work assignment.
enum class XStorage { Global, DistributedShared };

class DsmSpmvPlan {
public:
    int clusterBlocks() const { return cluster_blocks_; }
    int gridBlocks() const { return grid_blocks_; }
    int warpsPerBlock() const { return threads_ / 32; }
    int activeClusters() const { return active_clusters_; }
    std::size_t sharedBytesPerBlock() const { return shared_bytes_; }
    XStorage storage() const { return storage_; }
private:
    friend DsmSpmvPlan makeDsmSpmvPlan(const AtBsr<double, int32_t>&, int, XStorage, int, bool, int);
    friend void launchDsmSpmv(const DsmSpmvPlan&, const double*, double*, cudaStream_t);
    // The packed matrix must outlive the plan and must not be modified.
    const AtBsr<double, int32_t>* matrix_ = nullptr;
    const void* kernel_ = nullptr;
    int device_ = -1;
    int cluster_blocks_ = 1;
    int grid_blocks_ = 0;
    int active_clusters_ = 0;
    int threads_ = 128;
    std::size_t shared_bytes_ = 0;
    XStorage storage_ = XStorage::DistributedShared;
};

// cluster_blocks: 1 (local shared control), 2, 4 or 8.
// max_clusters=0 uses the occupancy-limited grid; a positive cap allows tests
// to force grid-stride reuse. Every cluster stages a complete copy of x.
// single_block_per_sm reserves >half the SM shared capacity to prevent multiple
// resident CTAs per SM. This is a scheduling experiment, not SM affinity.
// warps_per_block=0 uses the packed-matrix configuration; 2/4/8/16/32
// explicitly changes only the DSM/control launch, not the packed layout.
// Must be called outside stream capture; checks capacity and sets attributes.
DsmSpmvPlan makeDsmSpmvPlan(const AtBsr<double, int32_t>& matrix,
                          int cluster_blocks = 2,
                          XStorage storage = XStorage::DistributedShared,
                          int max_clusters = 0,
                          bool single_block_per_sm = false,
                          int warps_per_block = 0);

// Computes y=A*x, refreshing shared x on EVERY launch. x and y must not alias.
// Includes necessary output clearing for balanced tasks. Supports stream capture.
void launchDsmSpmv(const DsmSpmvPlan& plan, const double* x, double* y,
                   cudaStream_t stream = nullptr);

} // namespace atspmv
