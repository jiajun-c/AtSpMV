#pragma once
#include "mfem.hpp"
#include <memory>
#include <vector>

namespace atspmv_app {
// Reuse the symbolic union of A and B while only the coefficient values change.
// Inputs are finalized MFEM matrices (one stored entry per column in each row).
class SparseSum {
    std::unique_ptr<mfem::SparseMatrix> matrix_;
    std::vector<int> ap_, aj_, bp_, bj_;
    int *map_a_ = nullptr, *map_b_ = nullptr;
    void ReleaseMaps();
public:
    bool reused = false;
    ~SparseSum();
    mfem::SparseMatrix* Update(const mfem::SparseMatrix& a, const mfem::SparseMatrix& b,
                               double weight, bool reuse);
};
} // namespace atspmv_app
