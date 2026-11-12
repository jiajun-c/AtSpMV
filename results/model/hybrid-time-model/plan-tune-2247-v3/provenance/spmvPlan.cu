#include "spmvPlan.h"
#include "atspmvprod.h"
#include "unifiedSpmvModel.h"
#include "tileSelector.cuh"
#ifdef ATSPMV_PLAN_HAS_DSM
#include "detail/spmvPlanDsm.h"
#endif
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <map>
#include <cusparse.h>

namespace atspmv {
namespace {
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void checkSparse(cusparseStatus_t e) { if(e!=CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cuSPARSE status " + std::to_string(int(e))); }
struct Buffer {
    void* p = nullptr;
    ~Buffer() { if(p) cudaFree(p); }
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void allocate(size_t bytes) { check(cudaMalloc(&p,std::max(size_t(1),bytes))); }
    template<class T> T* get() const { return static_cast<T*>(p); }
};
struct Partition {
    Buffer longs, shorts;
    int nl = 0, ns = 0;
};
struct PackedOwner {
    AtBsr<double,int32_t> value;
    ~PackedOwner() {
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
// Same row/chunk assignment as the DSM backend, compiled for the normal
// architecture. One partial per CTA/row, followed by a separate gather.
template<int Rows>
__global__ void partialRows(const int* ptr,const int* col,const double* a,
                            const double* x,double* y,double* scratch,
                            const int* longs,int nl,const int* shorts,int ns,int parts) {
    const int long_blocks=((nl+Rows-1)/Rows)*parts;
    const int lane=threadIdx.x%32;
    if(blockIdx.x>=long_blocks) {
        int t=(blockIdx.x-long_blocks)*(blockDim.x/32)+threadIdx.x/32;
        int row=t<ns?shorts[t]:0;
        double sum=0;
        if(t<ns)for(int j=ptr[row]+lane;j<ptr[row+1];j+=32)sum+=a[j]*x[col[j]];
        for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
        if(t<ns&&lane==0)y[row]=sum;
        return;
    }
    __shared__ double sums[Rows][32];
    const int width=blockDim.x/Rows, local=threadIdx.x/width;
    const int task=blockIdx.x/parts*Rows+local, part=blockIdx.x%parts;
    const int tid=threadIdx.x%width, warp=tid/32;
    const int row=task<nl?longs[task]:0;
    const int begin=task<nl?ptr[row]:0, length=task<nl?ptr[row+1]-begin:0;
    const int first=begin+(long long)length*part/parts, last=begin+(long long)length*(part+1)/parts;
    double sum=0;
    for(int j=first+tid;j<last;j+=width)sum+=a[j]*x[col[j]];
    for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
    if(lane==0)sums[local][warp]=sum;
    __syncthreads();
    if(warp==0) {
        sum=lane<width/32?sums[local][lane]:0;
        for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
        if(lane==0&&task<nl)scratch[task*parts+part]=sum;
    }
}
template<int Width>
__global__ void vectorRows(const int* ptr,const int* col,const double* a,
                           const double* x,double* y,int rows) {
    constexpr int rows_per_block=256/Width;
    const auto row64=static_cast<long long>(blockIdx.x)*rows_per_block+threadIdx.x/Width;
    const int lane=threadIdx.x%Width;
    double sum=0;
    if(row64<rows) {
        const int row=static_cast<int>(row64);
        for(long long j=static_cast<long long>(ptr[row])+lane;j<ptr[row+1];j+=Width)sum+=a[j]*x[col[j]];
    }
    for(int d=Width/2;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d,Width);
    if(lane==0&&row64<rows)y[row64]=sum;
}
__global__ void finishRows(const double* scratch,double* y,const int* rows,int n,int parts) {
    int task=(blockIdx.x*blockDim.x+threadIdx.x)/32, lane=threadIdx.x%32;
    double sum=task<n&&lane<parts?scratch[task*parts+lane]:0;
    for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
    if(task<n&&lane==0)y[rows[task]]=sum;
}
void validate(const LongRowConfig& c) {
    if((c.parts!=2&&c.parts!=4&&c.parts!=8&&c.parts!=16&&c.parts!=32)||
       (c.threads!=128&&c.threads!=256&&c.threads!=512&&c.threads!=1024)||
       (c.rows_per_group!=1&&c.rows_per_group!=4&&c.rows_per_group!=8)||
       c.threads/c.rows_per_group<32||c.threshold<0)
        throw std::invalid_argument("invalid long-row configuration");
}
std::vector<LongRowConfig> shortlist() {
    std::vector<LongRowConfig> result;
    for(int threshold:{1024,4096})for(auto c:std::vector<LongRowConfig>{
        {8,128,1,0},{8,256,1,0},{4,256,1,0},{8,512,4,0},
        {8,128,4,0},{16,128,4,0},{32,128,1,0},
        {16,512,8,0},{32,256,4,0},{32,512,4,0},{32,256,8,0},{8,1024,8,0}}) {
        c.threshold=threshold;result.push_back(c);
    }
    result.push_back({2,256,1,0});
    return result;
}
}
struct SpmvPlanState {
    SpmvPlanInfo info;
    CSRMatrix<double,int32_t> csr;
    std::unique_ptr<Buffer> ptr, col, val;
    std::unique_ptr<Candidate> chosen;
    SpmvPlanState() {
        // CSRMatrix's legacy constructor initializes pointers only. Do not
        // leave optional metadata indeterminate in the plan-owned descriptor.
        csr.nnz=csr.nrows=csr.ncols=csr.FixLen=csr.minLen=csr.maxLen=csr.nonzeroRows=0;
        csr.useFixLen=csr.useColInt16=false;csr.stand=csr.avg_nnz=0;
    }
    ~SpmvPlanState() {
        int previous=-1;cudaGetDevice(&previous);
        if(info.device>=0&&previous!=info.device)cudaSetDevice(info.device);
        chosen.reset();ptr.reset();col.reset();val.reset();
        if(previous>=0&&previous!=info.device)cudaSetDevice(previous);
    }
};
namespace {
void launch(SpmvPlanState& s,Candidate& c,const double* x,double* y,cudaStream_t stream) {
    auto& a=s.csr;
    if(!a.nrows)return;
    if(!a.nnz) {check(cudaMemsetAsync(y,0,size_t(a.nrows)*sizeof(double),stream));return;}
    if(c.description.backend==SpmvBackend::AtBsr) {
        AtSpmvProduct(c.packed->value,const_cast<double*>(x),y,1,0,stream);
    } else if(c.description.backend==SpmvBackend::CsrVector) {
        const int width=c.description.vector_width;
        const int blocks=static_cast<int>((static_cast<long long>(a.nrows)+256/width-1)/(256/width));
#define VECTOR(W) vectorRows<W><<<blocks,256,0,stream>>>(a.row_ptr,a.col_idx,a.val,x,y,a.nrows)
        switch(width) {case 1:VECTOR(1);break;case 2:VECTOR(2);break;case 4:VECTOR(4);break;case 8:VECTOR(8);break;case 16:VECTOR(16);break;case 32:VECTOR(32);break;default:throw std::logic_error("invalid CSR vector width");}
#undef VECTOR
    } else if(c.description.backend==SpmvBackend::CuSparse) {
        const double alpha=1, beta=0;
        checkSparse(cusparseSetStream(c.sparse_handle,stream));
        checkSparse(cusparseDnVecSetValues(c.sparse_x,const_cast<double*>(x)));
        checkSparse(cusparseDnVecSetValues(c.sparse_y,y));
        checkSparse(cusparseSpMV(c.sparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,c.sparse_matrix,
                                c.sparse_x,&beta,c.sparse_y,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,c.scratch.p));
    } else {
        const auto cfg=c.description.long_rows;
        auto& p=*c.partition;
        if(c.description.backend==SpmvBackend::HybridDsmAtBsr) {
#ifdef ATSPMV_PLAN_HAS_DSM
            detail::launchPlanHybrid(a,c.packed->value,x,y,p.longs.get<int>(),p.nl,p.shorts.get<int>(),p.ns,cfg,stream);
#else
            throw std::logic_error("DSM backend not built");
#endif
        } else if(c.description.backend==SpmvBackend::CsrDsmVector) {
#ifdef ATSPMV_PLAN_HAS_DSM
            detail::launchPlanDsmVector(a,x,y,p.longs.get<int>(),p.nl,p.shorts.get<int>(),p.ns,cfg,c.description.vector_width,stream);
#else
            throw std::logic_error("DSM backend not built");
#endif
        } else if(c.description.backend==SpmvBackend::CsrDsm) {
#ifdef ATSPMV_PLAN_HAS_DSM
            detail::launchPlanDsm(a,x,y,p.longs.get<int>(),p.nl,p.shorts.get<int>(),p.ns,cfg,stream);
#else
            throw std::logic_error("DSM backend not built");
#endif
        } else {
            int blocks=((p.nl+cfg.rows_per_group-1)/cfg.rows_per_group)*cfg.parts+
                       (p.ns+cfg.threads/32-1)/(cfg.threads/32);
#define PARTIAL(G) partialRows<G><<<blocks,cfg.threads,0,stream>>>(a.row_ptr,a.col_idx,a.val,x,y,c.scratch.get<double>(),p.longs.get<int>(),p.nl,p.shorts.get<int>(),p.ns,cfg.parts)
            switch(cfg.rows_per_group) {case 1:PARTIAL(1);break;case 4:PARTIAL(4);break;case 8:PARTIAL(8);break;}
#undef PARTIAL
            if(p.nl)finishRows<<<(p.nl+7)/8,256,0,stream>>>(c.scratch.get<double>(),y,p.longs.get<int>(),p.nl,cfg.parts);
        }
    }
    check(cudaGetLastError());
}
}
SpmvPlan::SpmvPlan() = default;
SpmvPlan::~SpmvPlan() = default;
SpmvPlan::SpmvPlan(SpmvPlan&&) noexcept = default;
SpmvPlan& SpmvPlan::operator=(SpmvPlan&&) noexcept = default;
const SpmvPlanInfo& SpmvPlan::info() const {
    if(!state_)throw std::logic_error("empty or moved-from SpmvPlan");
    return state_->info;
}
const char* spmvBackendName(SpmvBackend b) {
    switch(b) {case SpmvBackend::Auto:return "auto";case SpmvBackend::AtBsr:return "atbsr";
    case SpmvBackend::CsrDsm:return "csr_dsm";case SpmvBackend::CsrTwoStage:return "csr_two_stage";
    case SpmvBackend::HybridDsmAtBsr:return "hybrid_dsm_atbsr";case SpmvBackend::CuSparse:return "cusparse";case SpmvBackend::CsrVector:return "csr_vector";case SpmvBackend::CsrDsmVector:return "csr_dsm_vector";}
    return "invalid";
}
SpmvPlan makeSpmvPlan(const CSRMatrix<double,int32_t>& input,const SpmvPlanOptions& o,cudaStream_t stream) {
    const auto started=std::chrono::steady_clock::now();
    if(o.backend!=SpmvBackend::Auto&&o.backend!=SpmvBackend::AtBsr&&o.backend!=SpmvBackend::CsrDsm&&o.backend!=SpmvBackend::CsrTwoStage&&o.backend!=SpmvBackend::HybridDsmAtBsr&&o.backend!=SpmvBackend::CuSparse&&o.backend!=SpmvBackend::CsrVector&&o.backend!=SpmvBackend::CsrDsmVector)
        throw std::invalid_argument("unknown SpMV backend");
    if(o.tile_index < -1||o.tile_index>5||o.balanced < -1||o.balanced>1||
       o.warmup_batches<0||o.warmup_batches>100||o.measurement_batches<1||o.measurement_batches>100||o.calls_per_batch<1||o.calls_per_batch>1024||
       !std::isfinite(o.hybrid_min_time_reduction)||o.hybrid_min_time_reduction<0||o.hybrid_min_time_reduction>=1)
        throw std::invalid_argument("invalid plan options");
    validate(o.long_rows);for(const auto& cfg:o.tuning_long_rows)validate(cfg);
    if(o.vector_width!=1&&o.vector_width!=2&&o.vector_width!=4&&o.vector_width!=8&&o.vector_width!=16&&o.vector_width!=32)throw std::invalid_argument("invalid CSR vector width");
    if(input.nrows<0||input.nrows>INT_MAX-1024||input.ncols<0||input.nnz<0||
       !input.row_ptr||(input.nnz&&(!input.val||!input.col_idx||!input.ncols)))
        throw std::invalid_argument("invalid CSR dimensions or pointers");
    cudaStreamCaptureStatus capture;check(cudaStreamIsCapturing(stream,&capture));
    if(capture!=cudaStreamCaptureStatusNone)throw std::invalid_argument("plan construction must be outside capture");
    check(cudaStreamSynchronize(stream));
    SpmvPlan plan;plan.state_=std::make_unique<SpmvPlanState>();auto& s=*plan.state_;
    check(cudaGetDevice(&s.info.device));
#ifdef ATSPMV_PLAN_HAS_DSM
    int supported=0;check(cudaDeviceGetAttribute(&supported,cudaDevAttrClusterLaunch,s.info.device));
    s.info.dsm_available=supported!=0;
#endif
    if((o.backend==SpmvBackend::CsrDsm||o.backend==SpmvBackend::HybridDsmAtBsr||o.backend==SpmvBackend::CsrDsmVector)&&!s.info.dsm_available)
        throw std::invalid_argument("DSM requires an enabled SM90 backend and cluster-capable device");
    if(!o.tune&&(o.backend==SpmvBackend::CsrDsm||o.backend==SpmvBackend::HybridDsmAtBsr||o.backend==SpmvBackend::CsrDsmVector)&&o.long_rows.parts>8)
        throw std::invalid_argument("DSM supports cluster sizes 2, 4, 8");
    if(o.backend==SpmvBackend::HybridDsmAtBsr&&o.balanced==1)
        throw std::invalid_argument("hybrid short rows currently require unbalanced AtBSR");
    std::vector<int> ptr(size_t(input.nrows)+1),cols(input.nnz);
    check(cudaMemcpy(ptr.data(),input.row_ptr,ptr.size()*sizeof(int),cudaMemcpyDeviceToHost));
    if(ptr.front()!=0||ptr.back()!=input.nnz||!std::is_sorted(ptr.begin(),ptr.end()))
        throw std::invalid_argument("invalid CSR offsets");
    if(input.nnz)check(cudaMemcpy(cols.data(),input.col_idx,cols.size()*sizeof(int),cudaMemcpyDeviceToHost));
    for(int c:cols)if(c<0||c>=input.ncols)throw std::invalid_argument("CSR column out of range");
    s.info.rows=s.csr.nrows=input.nrows;s.info.cols=s.csr.ncols=input.ncols;s.info.nnz=s.csr.nnz=input.nnz;
    s.ptr=std::make_unique<Buffer>();s.col=std::make_unique<Buffer>();s.val=std::make_unique<Buffer>();
    s.ptr->allocate(ptr.size()*sizeof(int));s.col->allocate(cols.size()*sizeof(int));s.val->allocate(size_t(input.nnz)*sizeof(double));
    s.csr.row_ptr=s.ptr->get<int>();s.csr.col_idx=s.col->get<int>();s.csr.val=s.val->get<double>();
    check(cudaMemcpy(s.csr.row_ptr,ptr.data(),ptr.size()*sizeof(int),cudaMemcpyHostToDevice));
    if(input.nnz) {
        check(cudaMemcpy(s.csr.col_idx,cols.data(),cols.size()*sizeof(int),cudaMemcpyHostToDevice));
        check(cudaMemcpy(s.csr.val,input.val,size_t(input.nnz)*sizeof(double),cudaMemcpyDeviceToDevice));
    }
    std::vector<std::unique_ptr<Candidate>> candidates;
    std::vector<double> host_values;
    // Partitions depend only on the matrix and threshold. Reuse immutable
    // row lists across all CSR and hybrid candidates during initialization.
    std::map<int,std::shared_ptr<Partition>> partitions;
    // Threshold partitions are nested, so equal long-row counts imply the same
    // row set. Share its packing across thresholds and DSM configurations.
    std::map<std::pair<int,int>,std::shared_ptr<PackedOwner>> hybrid_packing;
    auto add_impl=[&](SpmvBackend backend,int tile,int balance,LongRowConfig cfg,int vector_width) {
        if((backend==SpmvBackend::CsrDsm||backend==SpmvBackend::HybridDsmAtBsr||backend==SpmvBackend::CsrDsmVector)&&(!s.info.dsm_available||cfg.parts>8))return;
        auto c=std::make_unique<Candidate>();c->description.backend=backend;c->description.long_rows=cfg;
        c->description.vector_width=(backend==SpmvBackend::CsrVector||backend==SpmvBackend::CsrDsmVector)?vector_width:0;
        if(backend==SpmvBackend::AtBsr) {
            CSRToAtBSRProduct(s.csr,c->packed->value,tile,balance);
            int index=0;while((1<<index)!=c->packed->value.TILE_M)++index;
            c->description.tile_index=index;c->description.balanced=c->packed->value.useBalance;
        } else if(backend==SpmvBackend::CuSparse && input.nnz) {
            // Descriptors own no vector storage; the temporary vectors only
            // establish valid dimensions/pointers for workspace preparation.
            Buffer x,y;x.allocate(size_t(input.ncols)*sizeof(double));y.allocate(size_t(input.nrows)*sizeof(double));
            checkSparse(cusparseCreate(&c->sparse_handle));
            checkSparse(cusparseSetStream(c->sparse_handle,stream));
            checkSparse(cusparseCreateCsr(&c->sparse_matrix,input.nrows,input.ncols,input.nnz,s.csr.row_ptr,s.csr.col_idx,s.csr.val,
                                         CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
            checkSparse(cusparseCreateDnVec(&c->sparse_x,input.ncols,x.p,CUDA_R_64F));
            checkSparse(cusparseCreateDnVec(&c->sparse_y,input.nrows,y.p,CUDA_R_64F));
            const double alpha=1,beta=0;size_t bytes=0;
            checkSparse(cusparseSpMV_bufferSize(c->sparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,c->sparse_matrix,c->sparse_x,&beta,c->sparse_y,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,&bytes));
            c->scratch.allocate(bytes);
            checkSparse(cusparseSpMV_preprocess(c->sparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,c->sparse_matrix,c->sparse_x,&beta,c->sparse_y,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,c->scratch.p));
            check(cudaStreamSynchronize(stream));
        } else if(backend!=SpmvBackend::CuSparse&&backend!=SpmvBackend::CsrVector) {
            auto found_partition=partitions.find(cfg.threshold);
            std::vector<int> longs,shorts;
            if(found_partition!=partitions.end()) c->partition=found_partition->second;
            else {
                c->partition=std::make_shared<Partition>();auto& new_partition=*c->partition;
                for(int row=0;row<input.nrows;++row)(ptr[row+1]-ptr[row]>=cfg.threshold?longs:shorts).push_back(row);
                std::stable_sort(longs.begin(),longs.end(),[&](int a,int b){return ptr[a+1]-ptr[a]>ptr[b+1]-ptr[b];});
                new_partition.nl=longs.size();new_partition.ns=shorts.size();
                if((static_cast<long long>(new_partition.nl)+cfg.rows_per_group)*cfg.parts+new_partition.ns>INT_MAX)
                    throw std::invalid_argument("long-row work exceeds 32-bit indexing");
                new_partition.longs.allocate(longs.size()*sizeof(int));new_partition.shorts.allocate(shorts.size()*sizeof(int));
                if(new_partition.nl)check(cudaMemcpy(new_partition.longs.p,longs.data(),longs.size()*sizeof(int),cudaMemcpyHostToDevice));
                if(new_partition.ns)check(cudaMemcpy(new_partition.shorts.p,shorts.data(),shorts.size()*sizeof(int),cudaMemcpyHostToDevice));
                partitions.emplace(cfg.threshold,c->partition);
            }
            auto& p=*c->partition;
            if((static_cast<long long>(p.nl)+cfg.rows_per_group)*cfg.parts+p.ns>INT_MAX)
                throw std::invalid_argument("long-row work exceeds 32-bit indexing");
            if(backend==SpmvBackend::CsrTwoStage)c->scratch.allocate(size_t(p.nl)*cfg.parts*sizeof(double));
            if(backend==SpmvBackend::HybridDsmAtBsr) {
              const auto key=std::make_pair(p.nl,tile);
              auto found=hybrid_packing.find(key);
              if(found!=hybrid_packing.end())c->packed=found->second;
              else {
                // The row IDs were created by an earlier candidate when a
                // partition is shared; reconstruct the host short list for
                // packing without copying the shared device list back.
                if(found_partition!=partitions.end())
                    for(int row=0;row<input.nrows;++row)if(ptr[row+1]-ptr[row]<cfg.threshold)shorts.push_back(row);
                if(host_values.empty()&&input.nnz) {host_values.resize(input.nnz);check(cudaMemcpy(host_values.data(),input.val,size_t(input.nnz)*sizeof(double),cudaMemcpyDeviceToHost));}
                std::vector<int> shortptr(1,0),shortcols;std::vector<double> shortvals;
                for(int row:shorts) {
                    shortcols.insert(shortcols.end(),cols.begin()+ptr[row],cols.begin()+ptr[row+1]);
                    shortvals.insert(shortvals.end(),host_values.begin()+ptr[row],host_values.begin()+ptr[row+1]);
                    shortptr.push_back(static_cast<int>(shortcols.size()));
                }
                Buffer dptr,dcols,dvals;dptr.allocate(shortptr.size()*sizeof(int));dcols.allocate(shortcols.size()*sizeof(int));dvals.allocate(shortvals.size()*sizeof(double));
                check(cudaMemcpy(dptr.p,shortptr.data(),shortptr.size()*sizeof(int),cudaMemcpyHostToDevice));
                if(!shortcols.empty()) {check(cudaMemcpy(dcols.p,shortcols.data(),shortcols.size()*sizeof(int),cudaMemcpyHostToDevice));check(cudaMemcpy(dvals.p,shortvals.data(),shortvals.size()*sizeof(double),cudaMemcpyHostToDevice));}
                CSRMatrix<double,int32_t> shortcsr{};shortcsr.nrows=p.ns;shortcsr.ncols=input.ncols;shortcsr.nnz=shortcols.size();
                shortcsr.row_ptr=dptr.get<int>();shortcsr.col_idx=dcols.get<int>();shortcsr.val=dvals.get<double>();
                CSRToAtBSRProduct(shortcsr,c->packed->value,tile,0);
                if(shortcsr.nnz) {
                    std::vector<int> recorder(p.ns);check(cudaMemcpy(recorder.data(),c->packed->value.recorder,recorder.size()*sizeof(int),cudaMemcpyDeviceToHost));
                    for(auto& row:recorder)row=shorts.at(row);
                    check(cudaMemcpy(c->packed->value.recorder,recorder.data(),recorder.size()*sizeof(int),cudaMemcpyHostToDevice));
                } else c->packed->value.brow=0;
                hybrid_packing.emplace(key,c->packed);
              }
                int index=0;while((1<<index)!=c->packed->value.TILE_M)++index;
                c->description.tile_index=index;c->description.balanced=0;
            }
        }
        candidates.push_back(std::move(c));
    };
    auto add=[&](SpmvBackend backend,int tile,int balance,LongRowConfig cfg,int vector_width=0) {
        try { add_impl(backend,tile,balance,cfg,vector_width); }
        catch(const std::overflow_error& e) {
            // A forced tile can exceed AtBSR's 32-bit layout limits while
            // other layouts are valid. Preserve explicit untuned errors and
            // don't hide unrelated arithmetic or allocation failures.
            if(!o.tune||std::string(e.what())!="no tile layout fits the current 32-bit kernel offsets")throw;
            s.info.skipped_candidates.push_back(std::string(spmvBackendName(backend))+" tile="+std::to_string(tile)+" balance="+std::to_string(balance)+": "+e.what());
        }
    };
    size_t winner=0;
    if(!o.tune||!input.nnz) {
        add(o.backend==SpmvBackend::Auto?SpmvBackend::AtBsr:o.backend,o.tile_index,o.balanced,o.long_rows,o.vector_width);
        if(o.model_select&&o.backend==SpmvBackend::Auto&&input.nnz&&o.tile_index==-1&&o.balanced==-1&&s.info.dsm_available) {
            cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,s.info.device));
            bool model_env=false;
            for(const char* name:{"ATSPMV_MODEL_CACHE","ATSPMV_MODEL_MEMORY_GBPS","ATSPMV_MODEL_BW_EFFICIENCY","ATSPMV_MODEL_TILE_CYCLES","ATSPMV_MODEL_WARP_CYCLES","ATSPMV_MODEL_SHUFFLE_CYCLES","ATSPMV_MODEL_ATOMIC_NS","ATSPMV_MODEL_ATOMIC_SERIAL_NS","ATSPMV_MODEL_L2_MULTIPLIER","ATSPMV_MODEL_CACHE_FRACTION","ATSPMV_MODEL_CACHE_TRANSITION","ATSPMV_MODEL_LAUNCH_NS","ATSPMV_MODEL_X_BYTES","ATSPMV_MODEL_SATURATION_WARPS","ATSPMV_MODEL_TILE_NS"})
                if(std::getenv(name))model_env=true;
            if(prop.major==9&&prop.minor==0&&prop.multiProcessorCount==114&&prop.l2CacheSize==52428800&&
               std::string(prop.name)=="NVIDIA H100 PCIe"&&input.nnz<=1000000&&input.nrows<=349651&&input.ncols<=1785345&&
               !model_env&&kernelConfigFromEnvironment().baseline()) {
                const auto f1024=collectHybridModelFeatures(ptr,input.ncols,1024);
                const auto f4096=collectHybridModelFeatures(ptr,input.ncols,4096);
                s.info.model_reference_us=candidates[0]->packed->value.predictedTimeUs;
                s.info.model_reference_calibrated=candidates[0]->packed->value.modelCalibrated;
                const SpmvBackend backends[]={SpmvBackend::AtBsr,SpmvBackend::AtBsr,SpmvBackend::CsrDsm,SpmvBackend::CsrTwoStage,SpmvBackend::CuSparse,SpmvBackend::HybridDsmAtBsr,SpmvBackend::HybridDsmAtBsr,SpmvBackend::HybridDsmAtBsr};
                const int tiles[]={3,0,-1,-1,-1,3,5,5},balances[]={0,1,-1,-1,-1,0,0,0};
                const int feature_tiles[]={3,0,3,3,3,3,5,5};
                const LongRowConfig configs[]={{8,128,1,1024},{8,128,1,1024},{8,256,1,1024},{8,256,1,1024},{8,128,1,1024},{8,256,1,1024},{8,256,1,1024},{8,128,1,4096}};
                int best=-1;double best_ratio=1.;
                for(int i=0;i<8;++i) {
                    const auto& f=i==7?f4096:f1024;
                    if(i>=5&&!unifiedHybridEligible(f))continue;
                    const double ratio=predictUnifiedRatio(i,unifiedModelInputs(f,feature_tiles[i]));
                    s.info.model_predictions.push_back({backends[i],tiles[i],balances[i],configs[i],ratio,ratio*s.info.model_reference_us});
                    if(i<5&&ratio<0.95&&ratio<best_ratio) {best=i;best_ratio=ratio;}
                }
                const double hybrid_gate=1.-std::max(0.10,o.hybrid_min_time_reduction);
                int best_hybrid=-1;double hybrid_ratio=best_ratio*hybrid_gate;
                for(int i=5;i<8;++i) {
                    const auto& f=i==7?f4096:f1024;if(!unifiedHybridEligible(f))continue;
                    const double ratio=predictUnifiedRatio(i,unifiedModelInputs(f,feature_tiles[i]));
                    if(ratio<hybrid_ratio){best_hybrid=i;hybrid_ratio=ratio;}
                }
                if(best_hybrid>=0)best=best_hybrid;
                if(best>=0) {
                    try {add(backends[best],tiles[best],balances[best],configs[best]);winner=candidates.size()-1;}
                    catch(const std::overflow_error& e) {
                        if(std::string(e.what())!="no tile layout fits the current 32-bit kernel offsets")throw;
                        s.info.skipped_candidates.push_back(std::string("model candidate: ")+e.what());
                    }
                }
                s.info.model_used=true;
            }
        }
    } else {
        if(o.backend==SpmvBackend::Auto||o.backend==SpmvBackend::AtBsr)
            for(int t=0;t<6;++t)for(int b=0;b<2;++b)
                if((o.tile_index<0||t==o.tile_index)&&(o.balanced<0||b==o.balanced))add(SpmvBackend::AtBsr,t,b,{});
        if(o.backend==SpmvBackend::Auto||o.backend==SpmvBackend::CuSparse)
            add(SpmvBackend::CuSparse,-1,-1,{});
        if(o.backend==SpmvBackend::Auto||o.backend==SpmvBackend::CsrVector)
            for(int width:{1,2,4,8,16,32})add(SpmvBackend::CsrVector,-1,-1,{},width);
        if(s.info.dsm_available&&(o.backend==SpmvBackend::Auto||o.backend==SpmvBackend::CsrDsmVector)) {
            bool has_long=false;for(int row=0;row<input.nrows;++row)has_long|=ptr[row+1]-ptr[row]>=1024;
            if(o.backend==SpmvBackend::CsrDsmVector) {
                const auto vector_configs=o.tuning_long_rows.empty()?std::vector<LongRowConfig>{o.long_rows}:o.tuning_long_rows;
                for(auto cfg:vector_configs)for(int width:{1,2,4,8,16,32})add(SpmvBackend::CsrDsmVector,-1,-1,cfg,width);
            } else if(has_long)for(int width:{4,8,16,32})add(SpmvBackend::CsrDsmVector,-1,-1,{8,256,1,1024},width);
        }
        const auto configs=o.tuning_long_rows.empty()?shortlist():o.tuning_long_rows;
        for(auto backend:{SpmvBackend::CsrDsm,SpmvBackend::CsrTwoStage})
            if(o.backend==SpmvBackend::Auto||o.backend==backend)for(auto cfg:configs)add(backend,-1,-1,cfg);
        if((o.backend==SpmvBackend::Auto||o.backend==SpmvBackend::HybridDsmAtBsr)&&o.balanced!=1&&s.info.dsm_available) {
            auto hybrid_configs=configs;
            int max_row_length=0;bool has_medium_128=false,has_medium_256=false;
            for(int row=0;row<input.nrows;++row) {
                int length=ptr[row+1]-ptr[row];max_row_length=std::max(max_row_length,length);
                has_medium_128 |= length>=128 && length<1024;
                has_medium_256 |= length>=256 && length<1024;
            }
            if(o.tuning_long_rows.empty()) {
                if(has_medium_128) {
                    hybrid_configs.push_back({4,256,4,128});
                    hybrid_configs.push_back({8,256,4,128});
                }
                if(has_medium_256)hybrid_configs.push_back({4,256,4,256});
            }
            for(auto cfg:hybrid_configs)if(cfg.parts<=8)for(int tile=0;tile<6;++tile) {
                // When no row reaches the threshold, all DSM work disappears.
                if((o.backend!=SpmvBackend::Auto||!o.tuning_long_rows.empty()||cfg.threshold<=max_row_length)&&
                   (o.tile_index<0||o.tile_index==tile))add(SpmvBackend::HybridDsmAtBsr,tile,0,cfg);
            }
        }
    }
    if(candidates.empty())throw std::invalid_argument("no supported tuning candidates");
    if(o.tune&&input.nnz) {
        // Legacy/default streams cannot be captured. Use a private stream for
        // all measurement, after synchronizing the caller's input stream above.
        struct TuningStream {
            cudaStream_t value=nullptr;
            ~TuningStream() { if(value)cudaStreamDestroy(value); }
        } tuning;
        check(cudaStreamCreateWithFlags(&tuning.value,cudaStreamNonBlocking));
        Buffer x,y;x.allocate(size_t(input.ncols)*sizeof(double));y.allocate(size_t(input.nrows)*sizeof(double));
        std::vector<double> host_x(input.ncols);
        for(int i=0;i<input.ncols;++i)host_x[i]=((i*23LL)%97-48)/61.;
        check(cudaMemcpy(x.p,host_x.data(),host_x.size()*sizeof(double),cudaMemcpyHostToDevice));
        for(auto& c:candidates) {
            launch(s,*c,x.get<double>(),y.get<double>(),tuning.value);check(cudaStreamSynchronize(tuning.value));
            check(cudaStreamBeginCapture(tuning.value,cudaStreamCaptureModeThreadLocal));
            try {
                for(int k=0;k<o.calls_per_batch;++k)launch(s,*c,x.get<double>(),y.get<double>(),tuning.value);
                check(cudaStreamEndCapture(tuning.value,&c->graph));
            } catch(...) {
                cudaStreamCaptureStatus status;
                if(cudaStreamIsCapturing(tuning.value,&status)==cudaSuccess&&status!=cudaStreamCaptureStatusNone) {
                    cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(tuning.value,&abandoned);
                    if(abandoned)cudaGraphDestroy(abandoned);
                }
                throw;
            }
            check(cudaGraphInstantiate(&c->executable,c->graph,nullptr,nullptr,0));
        }
        struct Events {cudaEvent_t start=nullptr,stop=nullptr;~Events(){if(start)cudaEventDestroy(start);if(stop)cudaEventDestroy(stop);}} events;
        check(cudaEventCreate(&events.start));check(cudaEventCreate(&events.stop));
        std::vector<size_t> order(candidates.size());std::iota(order.begin(),order.end(),0);std::mt19937 random(20261027);
        for(int batch=-o.warmup_batches;batch<o.measurement_batches;++batch) {
            std::shuffle(order.begin(),order.end(),random);
            for(size_t i:order) {
                auto& c=*candidates[i];
                check(cudaGraphLaunch(c.executable,tuning.value)); // Warm this candidate's working set.
                check(cudaEventRecord(events.start,tuning.value));check(cudaGraphLaunch(c.executable,tuning.value));
                check(cudaEventRecord(events.stop,tuning.value));check(cudaEventSynchronize(events.stop));
                float ms;check(cudaEventElapsedTime(&ms,events.start,events.stop));
                if(batch>=0)c.samples.push_back(ms*1000./o.calls_per_batch);
            }
        }
        for(size_t i=0;i<candidates.size();++i) {
            auto& c=*candidates[i];std::sort(c.samples.begin(),c.samples.end());
            c.description.median_us=(c.samples[(c.samples.size()-1)/2]+c.samples[c.samples.size()/2])/2;
            s.info.timings.push_back(c.description);
            if(c.description.median_us<candidates[winner]->description.median_us)winner=i;
        }
        if(o.backend==SpmvBackend::Auto) {
            size_t nonhybrid=candidates.size(), hybrid=candidates.size();
            for(size_t i=0;i<candidates.size();++i) {
                const bool is_hybrid=candidates[i]->description.backend==SpmvBackend::HybridDsmAtBsr||
                                     candidates[i]->description.backend==SpmvBackend::CsrDsmVector;
                if(is_hybrid&&!candidates[i]->partition->nl)continue;
                size_t& best=is_hybrid?hybrid:nonhybrid;
                if(best==candidates.size()||candidates[i]->description.median_us<candidates[best]->description.median_us)best=i;
            }
            if(nonhybrid!=candidates.size()) {
                winner=nonhybrid;
                if(hybrid!=candidates.size() && candidates[hybrid]->description.median_us<=candidates[nonhybrid]->description.median_us*(1-o.hybrid_min_time_reduction))winner=hybrid;
            }
        }
        s.info.measured=true;
        // Captured graphs reference private tuning X/Y; destroy them before
        // freeing those buffers. Execution uses direct launches on the winner.
        for(auto& c:candidates) {
            check(cudaGraphExecDestroy(c->executable));c->executable=nullptr;
            check(cudaGraphDestroy(c->graph));c->graph=nullptr;
        }
    }
    s.chosen=std::move(candidates[winner]);candidates.clear();
    // Account for cleanup of cached unselected data in reported setup time.
    hybrid_packing.clear();partitions.clear();
    const auto& chosen=s.chosen->description;
    s.info.backend=chosen.backend;s.info.tile_index=chosen.tile_index;s.info.balanced=chosen.balanced;s.info.long_rows=chosen.long_rows;
    s.info.vector_width=chosen.vector_width;
    if(s.info.backend==SpmvBackend::AtBsr) {s.ptr.reset();s.col.reset();s.val.reset();s.csr.row_ptr=nullptr;s.csr.col_idx=nullptr;s.csr.val=nullptr;}
    s.info.setup_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count();
    return plan;
}
void executeSpmv(const SpmvPlan& plan,const double* x,double* y,cudaStream_t stream) {
    if(!plan.state_)throw std::logic_error("empty or moved-from SpmvPlan");
    auto& s=*plan.state_;int device;check(cudaGetDevice(&device));
    if(device!=s.info.device)throw std::invalid_argument("SpmvPlan used on a different CUDA device");
    if(!s.info.rows)return;
    if(!y||(s.info.nnz&&!x))throw std::invalid_argument("null SpMV vector");
    if(x) {
        const auto xp=reinterpret_cast<uintptr_t>(x),yp=reinterpret_cast<uintptr_t>(y);
        if(xp<yp+size_t(s.info.rows)*sizeof(double)&&yp<xp+size_t(s.info.cols)*sizeof(double))
            throw std::invalid_argument("overlapping SpMV vectors");
    }
    launch(s,*s.chosen,x,y,stream);
}
} // namespace atspmv
