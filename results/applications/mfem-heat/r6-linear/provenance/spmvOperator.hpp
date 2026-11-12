#pragma once
#include "mfem.hpp"
#include "spmvPlan.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

namespace atspmv_app {
using Clock = std::chrono::steady_clock;
inline double Seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
inline void Sync() {
    auto status = cudaDeviceSynchronize();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Statistics {
    std::string backend = "cusparse";
    double plan_seconds = 0;
    long long plans = 0, multiplies = 0, cg_iterations = 0, solves = 0;
    int max_rows = 0, max_nnz = 0;
};

// A serial MFEM operator: all vector kernels and Plan calls use CUDA stream 0.
// The native MFEM matrix also supplies the unchanged diagonal preconditioner.
class SpmvOperator : public mfem::Operator {
    const mfem::SparseMatrix* matrix_ = nullptr;
    Statistics& stats_;
    atspmv::SpmvPlan plan_;
public:
    explicit SpmvOperator(Statistics& stats) : stats_(stats) {}
    const mfem::SparseMatrix& Matrix() const { return *matrix_; }
    void SetMatrix(const mfem::SparseMatrix& matrix) {
        Sync();
        auto start = Clock::now();
        matrix_ = &matrix;
        height = matrix.Height(); width = matrix.Width();
        if (stats_.backend != "mfem") {
            CSRMatrix<double, int32_t> csr;
            csr.nrows = height; csr.ncols = width; csr.nnz = matrix.NumNonZeroElems();
            csr.row_ptr = const_cast<int*>(matrix.ReadI());
            csr.col_idx = const_cast<int*>(matrix.ReadJ());
            csr.val = const_cast<double*>(matrix.ReadData());
            atspmv::SpmvPlanOptions options;
            if (stats_.backend == "cusparse") options.backend = atspmv::SpmvBackend::CuSparse;
            else if (stats_.backend == "atbsr") options.backend = atspmv::SpmvBackend::AtBsr;
            else if (stats_.backend == "mixed") options.backend = atspmv::SpmvBackend::MixedTile;
            else if (stats_.backend == "tuned") {
                options.tune = true; options.fast_tune = true; options.enable_mixed_tile = true;
            } else throw std::invalid_argument("backend must be mfem, cusparse, atbsr, mixed, or tuned");
            plan_ = atspmv::makeSpmvPlan(csr, options);
            const auto& info = plan_.info();
            std::cout << "plan rows=" << height << " nnz=" << csr.nnz
                      << " backend=" << atspmv::spmvBackendName(info.backend)
                      << " tile=" << info.tile_index << " balanced=" << info.balanced
                      << " mixed_cap=" << info.mixed_tile_cap << '\n';
        }
        Sync();
        stats_.plan_seconds += Seconds(start);
        ++stats_.plans;
        stats_.max_rows = std::max(stats_.max_rows, height);
        stats_.max_nnz = std::max(stats_.max_nnz, matrix.NumNonZeroElems());
    }
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override {
        if (stats_.backend == "mfem") matrix_->Mult(x, y);
        else atspmv::executeSpmv(plan_, x.Read(), y.Write());
        ++stats_.multiplies;
    }
};

class Jacobi : public mfem::DSmoother {
public:
    void SetOperator(const mfem::Operator& op) override {
        if (auto wrapped = dynamic_cast<const SpmvOperator*>(&op))
            mfem::DSmoother::SetOperator(wrapped->Matrix());
        else mfem::DSmoother::SetOperator(op);
    }
};
} // namespace atspmv_app
