#pragma once
#include "mfem.hpp"
#include <memory>
#include <vector>

namespace atspmv_app {
// Exact comparison on the GPU avoids downloading new CSR indices every step.
class CsrStructure {
    int rows_ = -1, cols_ = -1, nnz_ = -1;
    int *ptr_ = nullptr, *col_ = nullptr, *different_ = nullptr;
    void Release();
public:
    CsrStructure() = default;
    CsrStructure(const CsrStructure&) = delete;
    CsrStructure& operator=(const CsrStructure&) = delete;
    ~CsrStructure();
    bool Matches(const mfem::SparseMatrix& matrix) const;
    void Remember(const mfem::SparseMatrix& matrix);
};
// Reuse the symbolic union of A and B while only the coefficient values change.
// Inputs are finalized MFEM matrices (one stored entry per column in each row).
class SparseSum {
    std::unique_ptr<mfem::SparseMatrix> matrix_;
    CsrStructure a_structure_, b_structure_;
    int *map_a_ = nullptr, *map_b_ = nullptr;
    void ReleaseMaps();
public:
    bool reused = false;
    ~SparseSum();
    mfem::SparseMatrix* Update(const mfem::SparseMatrix& a, const mfem::SparseMatrix& b,
                               double weight, bool reuse);
};
} // namespace atspmv_app
