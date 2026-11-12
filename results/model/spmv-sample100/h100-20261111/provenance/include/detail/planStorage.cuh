#pragma once

#include "atspmvprod.h"
#include "cudaError.h"
#include "spmvPlan.h"
#include <cusparse.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace atspmv::detail {

struct Mixed;
struct CompressedTileColumns;
struct GroupedTileColumns;

inline void planStorageCheck(cudaError_t error) { if(error!=cudaSuccess)throw CudaError(error); }
class SparseError : public std::runtime_error {
public:
    explicit SparseError(cusparseStatus_t e) : std::runtime_error("cuSPARSE status " + std::to_string(int(e))), code(e) {}
    cusparseStatus_t code;
};
inline void checkSparse(cusparseStatus_t e) { if(e!=CUSPARSE_STATUS_SUCCESS) throw SparseError(e); }

struct Buffer {
    void* p = nullptr;
    ~Buffer() { reset(); }
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void allocate(size_t bytes) { planStorageCheck(cudaMalloc(&p,std::max(size_t(1),bytes))); }
    void reset() { if(p) { cudaFree(p);p=nullptr; } }
    template<class T> T* get() const { return static_cast<T*>(p); }
};
struct Partition {
    Buffer longs, shorts;
    int nl = 0, ns = 0;
};
struct ShortPackingInput {
    Buffer ptr, col, val, lengths, indices;
    AtBSRProductStatsCache stats;
    CSRMatrix<double,int32_t> csr{};
    std::vector<int> rows;
    int long_rows = -1;
};
struct PackedOwner {
    AtBsr<double,int32_t> value;
    // Aliases keep the underlying packing alive without freeing it twice.
    std::shared_ptr<PackedOwner> backing;
    ~PackedOwner() {
        if(backing)return;
        for(void* p:{static_cast<void*>(value.blcVal),static_cast<void*>(value.blcIdx),
                    static_cast<void*>(value.blcIdxInt16),static_cast<void*>(value.blcPtr),
                    static_cast<void*>(value.recorder),static_cast<void*>(value.targetRowByWarp),
                    static_cast<void*>(value.warpBlcPtr)}) if(p) cudaFree(p);
    }
};
struct Candidate {
    SpmvCandidateTiming description;
    std::shared_ptr<PackedOwner> packed = std::make_shared<PackedOwner>();
    std::shared_ptr<Partition> partition;
    std::shared_ptr<Mixed> mixed;
    std::shared_ptr<CompressedTileColumns> compressed;
    std::shared_ptr<GroupedTileColumns> grouped;
    Buffer scratch;
    cusparseHandle_t sparse_handle = nullptr;
    cusparseSpMatDescr_t sparse_matrix = nullptr;
    cusparseDnVecDescr_t sparse_x = nullptr, sparse_y = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    std::vector<double> samples;
    ~Candidate() {
        if(executable) cudaGraphExecDestroy(executable);
        if(graph) cudaGraphDestroy(graph);
        if(sparse_x) cusparseDestroyDnVec(sparse_x);
        if(sparse_y) cusparseDestroyDnVec(sparse_y);
        if(sparse_matrix) cusparseDestroySpMat(sparse_matrix);
        if(sparse_handle) cusparseDestroy(sparse_handle);
    }
};

} // namespace atspmv::detail
