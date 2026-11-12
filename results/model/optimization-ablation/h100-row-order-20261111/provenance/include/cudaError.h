#pragma once
#include <cuda_runtime_api.h>
#include <stdexcept>

namespace atspmv {
// Preserve the CUDA status for selective recovery while retaining the
// existing std::runtime_error interface and diagnostic text.
class CudaError : public std::runtime_error {
public:
    explicit CudaError(cudaError_t code) : std::runtime_error(cudaGetErrorString(code)), code_(code) {}
    cudaError_t code() const noexcept { return code_; }
private:
    cudaError_t code_;
};
}
