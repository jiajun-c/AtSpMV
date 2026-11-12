#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace atspmv::detail {
struct RowNNZCalculator {
    const int32_t* row_ptr;

    explicit RowNNZCalculator(const int32_t* ptr) : row_ptr(ptr) {}

    __host__ __device__ int32_t operator()(int32_t row) const {
        return row_ptr[row + 1] - row_ptr[row];
    }
};
} // namespace atspmv::detail
