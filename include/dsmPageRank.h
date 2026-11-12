#pragma once
#include "dsmSpmv.h"
#include <memory>

namespace atspmv {
struct PageRankState;
// Owns scratch space, but not CSR values/indices or the dangling-node mask.
// Input CSR is A^T with each edge weighted by 1/out_degree(source).
// CSR and mask must remain alive until all launches finish, on the plan's device.
// Global-storage plans own a reusable workspace: do not launch the same plan
// concurrently on multiple streams. Create one plan per concurrent solve.
class PersistentPageRankPlan {
public:
    std::size_t sharedBytesPerBlock() const;
    int clusterBlocks() const;
private:
    std::shared_ptr<PageRankState> state_;
    friend PersistentPageRankPlan makePersistentPageRankPlan(
        const CSRMatrix<double,int32_t>&, const int*, bool, int, XStorage);
    friend void launchPersistentPageRank(const PersistentPageRankPlan&,
        const double*, double*, int, double, cudaStream_t);
};
PersistentPageRankPlan makePersistentPageRankPlan(const CSRMatrix<double,int32_t>& normalized_transpose,
    const int* dangling_mask, bool has_dangling, int cluster_blocks=4,
    XStorage storage=XStorage::DistributedShared);
// All iterations execute in one cluster. Includes initial x staging, dangling
// redistribution and final result store. Setup occurs outside stream capture.
void launchPersistentPageRank(const PersistentPageRankPlan&, const double* initial, double* output,
                             int iterations, double damping=0.85, cudaStream_t stream=nullptr);
} // namespace atspmv
