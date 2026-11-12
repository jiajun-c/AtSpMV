#include "spmvPlan.h"
#include "atspmvprod.h"
#include "detail/planModelSelection.h"
#include "detail/planTimingSelection.h"
#include "detail/planStorage.cuh"
#include "detail/mixedTile.cuh"
#include "detail/spmvPlanKernels.cuh"
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
void check(cudaError_t e) { if(e!=cudaSuccess) throw CudaError(e); }
using detail::SparseError;
using detail::checkSparse;
bool unifiedModelSupported(const CSRMatrix<double,int32_t>& input,int device,bool dsm_available) {
    if(!dsm_available||input.nnz>1000000||input.nrows>349651||input.ncols>1785345)return false;
    for(const char* name:{"ATSPMV_MODEL_CACHE","ATSPMV_MODEL_MEMORY_GBPS","ATSPMV_MODEL_BW_EFFICIENCY","ATSPMV_MODEL_TILE_CYCLES","ATSPMV_MODEL_WARP_CYCLES","ATSPMV_MODEL_SHUFFLE_CYCLES","ATSPMV_MODEL_ATOMIC_NS","ATSPMV_MODEL_ATOMIC_SERIAL_NS","ATSPMV_MODEL_L2_MULTIPLIER","ATSPMV_MODEL_CACHE_FRACTION","ATSPMV_MODEL_CACHE_TRANSITION","ATSPMV_MODEL_LAUNCH_NS","ATSPMV_MODEL_X_BYTES","ATSPMV_MODEL_SATURATION_WARPS","ATSPMV_MODEL_TILE_NS"})
        if(std::getenv(name))return false;
    if(!kernelConfigFromEnvironment().baseline())return false;
    cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,device));
    return prop.major==9&&prop.minor==0&&prop.multiProcessorCount==114&&prop.l2CacheSize==52428800&&std::string(prop.name)=="NVIDIA H100 PCIe";
}
using detail::Buffer;
using detail::Partition;
using detail::ShortPackingInput;
using detail::PackedOwner;
using detail::Candidate;
using detail::partialRows;
using detail::vectorRows;
using detail::finishRows;
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
    } else if(c.description.backend==SpmvBackend::MixedTile) {
        detail::launchMixed(*c.mixed,c.description.mixed_tile_cap,a.nrows,x,y,stream);
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
    case SpmvBackend::HybridDsmAtBsr:return "hybrid_dsm_atbsr";case SpmvBackend::CuSparse:return "cusparse";case SpmvBackend::CsrVector:return "csr_vector";case SpmvBackend::CsrDsmVector:return "csr_dsm_vector";case SpmvBackend::MixedTile:return "mixed_tile";}
    return "invalid";
}
SpmvPlan makeSpmvPlan(const CSRMatrix<double,int32_t>& input,const SpmvPlanOptions& o,cudaStream_t stream) {
    const auto started=std::chrono::steady_clock::now();
    auto can_retry=[&] {return o.retry_shared_storage_on_oom&&o.tune&&!o.fast_tune&&
        !o.share_atbsr_tuning_storage&&o.balanced==-1&&
        (o.backend==SpmvBackend::Auto||o.backend==SpmvBackend::AtBsr);};
    auto retry_with_shared=[&] {
        const auto pending=cudaGetLastError();
        if(pending!=cudaSuccess&&pending!=cudaErrorMemoryAllocation)throw CudaError(pending);
        auto retry=o;retry.share_atbsr_tuning_storage=true;retry.retry_shared_storage_on_oom=false;
        auto result=makeSpmvPlan(input,retry,stream);
        result.state_->info.oom_retry_shared_storage_used=true;
        result.state_->info.setup_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count();
        return result;
    };
    try {
    if(o.backend!=SpmvBackend::Auto&&o.backend!=SpmvBackend::AtBsr&&o.backend!=SpmvBackend::CsrDsm&&o.backend!=SpmvBackend::CsrTwoStage&&o.backend!=SpmvBackend::HybridDsmAtBsr&&o.backend!=SpmvBackend::CuSparse&&o.backend!=SpmvBackend::CsrVector&&o.backend!=SpmvBackend::CsrDsmVector&&o.backend!=SpmvBackend::MixedTile)
        throw std::invalid_argument("unknown SpMV backend");
    if(o.vector_model_select&&(!o.model_select||o.tune||o.backend!=SpmvBackend::Auto))
        throw std::invalid_argument("vector model requires untuned Auto with model_select");
    if(o.model_refine&&(!o.model_select||o.tune||o.backend!=SpmvBackend::Auto||o.tile_index!=-1||o.balanced!=-1))
        throw std::invalid_argument("model refinement requires untuned Auto with model_select and unforced tile/mode");
    if(o.model_fallback_tune&&(!o.model_select||o.tune||o.backend!=SpmvBackend::Auto))
        throw std::invalid_argument("model fallback tuning requires untuned Auto with model_select");
    if(o.tile_index < -1||o.tile_index>5||o.balanced < -1||o.balanced>1||
       o.warmup_batches<0||o.warmup_batches>100||o.measurement_batches<1||o.measurement_batches>100||o.calls_per_batch<1||o.calls_per_batch>1024||
       !std::isfinite(o.hybrid_min_time_reduction)||o.hybrid_min_time_reduction<0||o.hybrid_min_time_reduction>=1||
       !std::isfinite(o.vector_model_min_time_reduction)||o.vector_model_min_time_reduction<0||o.vector_model_min_time_reduction>=1)
        throw std::invalid_argument("invalid plan options");
    if(o.mixed_tile_cap!=-1&&o.mixed_tile_cap!=0&&o.mixed_tile_cap!=4&&o.mixed_tile_cap!=8&&o.mixed_tile_cap!=16)throw std::invalid_argument("invalid mixed tile cap");
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
    const bool try_model=o.model_select&&!o.tune&&o.backend==SpmvBackend::Auto&&input.nnz&&o.tile_index==-1&&o.balanced==-1;
    const bool model_supported=try_model&&unifiedModelSupported(input,s.info.device,s.info.dsm_available);
    if(try_model&&o.model_fallback_tune&&!model_supported) {
        auto fallback=o;fallback.tune=true;fallback.model_select=false;fallback.vector_model_select=false;fallback.model_refine=false;fallback.model_fallback_tune=false;
        auto tuned=makeSpmvPlan(input,fallback,stream);
        tuned.state_->info.model_fallback_tune_used=true;
        tuned.state_->info.setup_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count();
        return tuned;
    }
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
    std::shared_ptr<detail::Mixed> mixed_packing;
    // Partitions depend only on the matrix and threshold. Reuse immutable
    // row lists across all CSR and hybrid candidates during initialization.
    std::map<int,std::shared_ptr<Partition>> partitions;
    // Threshold partitions are nested, so equal long-row counts imply the same
    // row set. Share its packing across thresholds and DSM configurations.
    std::map<std::pair<int,int>,std::shared_ptr<PackedOwner>> hybrid_packing;
    std::map<int,std::shared_ptr<PackedOwner>> balanced_atbsr_packing;
    // Only the current partition's CSR and row ordering are retained. This
    // bounds the extra workspace even when many thresholds are considered.
    std::unique_ptr<ShortPackingInput> short_packing_input;
    Buffer sorted_lengths,sorted_indices;
    AtBSRProductStatsCache sorted_stats;
    const bool fast_shortlist=o.backend==SpmvBackend::Auto&&o.fast_tune&&o.tile_index<0&&o.balanced<0&&o.tuning_long_rows.empty();
    const bool reuse_atbsr_order=o.tune&&input.nnz&&
        (o.backend==SpmvBackend::AtBsr||o.backend==SpmvBackend::Auto)&&!fast_shortlist&&
        (o.tile_index<0||o.balanced<0);
    auto add_impl=[&](SpmvBackend backend,int tile,int balance,LongRowConfig cfg,int vector_width,int mixed_cap) {
        if((backend==SpmvBackend::CsrDsm||backend==SpmvBackend::HybridDsmAtBsr||backend==SpmvBackend::CsrDsmVector)&&(!s.info.dsm_available||cfg.parts>8))return;
        auto c=std::make_unique<Candidate>();c->description.backend=backend;c->description.long_rows=cfg;
        c->description.vector_width=(backend==SpmvBackend::CsrVector||backend==SpmvBackend::CsrDsmVector)?vector_width:0;
        if(backend==SpmvBackend::MixedTile) {
            if(!mixed_packing) {
                mixed_packing=std::make_shared<detail::Mixed>();
                if(host_values.empty()&&input.nnz) {host_values.resize(input.nnz);check(cudaMemcpy(host_values.data(),input.val,size_t(input.nnz)*sizeof(double),cudaMemcpyDeviceToHost));}
                detail::make_mixed(*mixed_packing,ptr,cols,host_values,input.ncols);
            }
            c->mixed=mixed_packing;c->description.mixed_tile_cap=mixed_cap;
        } else if(backend==SpmvBackend::AtBsr) {
            auto shared=balanced_atbsr_packing.find(tile);
            if(o.share_atbsr_tuning_storage&&balance==0&&shared!=balanced_atbsr_packing.end()) {
                c->packed->backing=shared->second;
                c->packed->value=shared->second->value;
                c->packed->value.useBalance=false;
                c->packed->value.warpCount=c->packed->value.brow;
            } else if(reuse_atbsr_order) {
                if(!sorted_lengths.p) {
                    sorted_lengths.allocate(size_t(input.nrows)*sizeof(int));
                    sorted_indices.allocate(size_t(input.nrows)*sizeof(int));
                    prepareAtBSRProductRowOrder(s.csr,sorted_lengths.get<int>(),sorted_indices.get<int>());
                }
                CSRToAtBSRProductFromRows(s.csr,c->packed->value,tile,balance,sorted_lengths.get<int>(),sorted_indices.get<int>(),&sorted_stats);
            } else CSRToAtBSRProduct(s.csr,c->packed->value,tile,balance);
            if(o.tune&&o.share_atbsr_tuning_storage&&tile>=0&&balance==1)
                balanced_atbsr_packing[tile]=c->packed;
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
                // Threshold partitions are nested, so equal long-row counts
                // identify the same short rows in the same original order.
                if(!short_packing_input||short_packing_input->long_rows!=p.nl) {
                    short_packing_input=std::make_unique<ShortPackingInput>();auto& cache=*short_packing_input;
                    cache.long_rows=p.nl;
                    if(found_partition!=partitions.end())
                        for(int row=0;row<input.nrows;++row)if(ptr[row+1]-ptr[row]<cfg.threshold)shorts.push_back(row);
                    cache.rows=std::move(shorts);
                    if(host_values.empty()&&input.nnz) {host_values.resize(input.nnz);check(cudaMemcpy(host_values.data(),input.val,size_t(input.nnz)*sizeof(double),cudaMemcpyDeviceToHost));}
                    std::vector<int> shortptr(1,0),shortcols;std::vector<double> shortvals;
                    for(int row:cache.rows) {
                        shortcols.insert(shortcols.end(),cols.begin()+ptr[row],cols.begin()+ptr[row+1]);
                        shortvals.insert(shortvals.end(),host_values.begin()+ptr[row],host_values.begin()+ptr[row+1]);
                        shortptr.push_back(static_cast<int>(shortcols.size()));
                    }
                    cache.ptr.allocate(shortptr.size()*sizeof(int));cache.col.allocate(shortcols.size()*sizeof(int));cache.val.allocate(shortvals.size()*sizeof(double));
                    check(cudaMemcpy(cache.ptr.p,shortptr.data(),shortptr.size()*sizeof(int),cudaMemcpyHostToDevice));
                    if(!shortcols.empty()) {check(cudaMemcpy(cache.col.p,shortcols.data(),shortcols.size()*sizeof(int),cudaMemcpyHostToDevice));check(cudaMemcpy(cache.val.p,shortvals.data(),shortvals.size()*sizeof(double),cudaMemcpyHostToDevice));}
                    cache.csr.nrows=p.ns;cache.csr.ncols=input.ncols;cache.csr.nnz=shortcols.size();
                    cache.csr.row_ptr=cache.ptr.get<int>();cache.csr.col_idx=cache.col.get<int>();cache.csr.val=cache.val.get<double>();
                    if(o.tune&&cache.csr.nnz) {
                        cache.lengths.allocate(size_t(p.ns)*sizeof(int));cache.indices.allocate(size_t(p.ns)*sizeof(int));
                        prepareAtBSRProductRowOrder(cache.csr,cache.lengths.get<int>(),cache.indices.get<int>());
                    }
                }
                auto& cache=*short_packing_input;
                if(cache.lengths.p) CSRToAtBSRProductFromRows(cache.csr,c->packed->value,tile,0,cache.lengths.get<int>(),cache.indices.get<int>(),&cache.stats);
                else CSRToAtBSRProduct(cache.csr,c->packed->value,tile,0);
                if(cache.csr.nnz) {
                    std::vector<int> recorder(p.ns);check(cudaMemcpy(recorder.data(),c->packed->value.recorder,recorder.size()*sizeof(int),cudaMemcpyDeviceToHost));
                    for(auto& row:recorder)row=cache.rows.at(row);
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
    auto add=[&](SpmvBackend backend,int tile,int balance,LongRowConfig cfg,int vector_width=0,int mixed_cap=0) {
        try { add_impl(backend,tile,balance,cfg,vector_width,mixed_cap); }
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
        add(o.backend==SpmvBackend::Auto?SpmvBackend::AtBsr:o.backend, o.tile_index, o.balanced,
            o.long_rows, o.vector_width, o.mixed_tile_cap);
        if(model_supported) {
            s.info.model_reference_us=candidates[0]->packed->value.predictedTimeUs;
            s.info.model_reference_calibrated=candidates[0]->packed->value.modelCalibrated;
            auto choice=detail::choosePlanModel(ptr,input.ncols,s.info.model_reference_us,o);
            s.info.model_predictions=std::move(choice.predictions);
            s.info.vector_model_used=choice.vector_model_used;
            if(choice.selected) {
                try {
                    add(choice.backend,choice.tile_index,choice.balanced,choice.long_rows,choice.vector_width);
                    winner=candidates.size()-1;
                } catch(const std::overflow_error& e) {
                    if(std::string(e.what())!="no tile layout fits the current 32-bit kernel offsets")throw;
                    s.info.skipped_candidates.push_back(std::string("model candidate: ")+e.what());
                }
            }
            s.info.model_used=true;
        }
    } else if(o.backend==SpmvBackend::Auto&&o.fast_tune&&o.tile_index<0&&o.balanced<0&&o.tuning_long_rows.empty()) {
        // Keep the model-selected AtBSR and all inexpensive CSR vector widths.
        // Other candidates are a bounded shortlist, not an exhaustive search.
        s.info.fast_tuning_used=true;
        add(SpmvBackend::AtBsr,-1,-1,{});
        add(SpmvBackend::CuSparse,-1,-1,{});
        for(int width:{1,2,4,8,16,32})add(SpmvBackend::CsrVector,-1,-1,{},width);
        add(SpmvBackend::CsrTwoStage,-1,-1,{8,256,1,1024});
        if(s.info.dsm_available) {
            add(SpmvBackend::CsrDsm,-1,-1,{8,256,1,1024});
            int maximum=0;for(int row=0;row<input.nrows;++row)maximum=std::max(maximum,ptr[row+1]-ptr[row]);
            if(maximum>=1024) {
                for(int width:{4,8,16,32})add(SpmvBackend::CsrDsmVector,-1,-1,{8,256,1,1024},width);
                for(int tile:{3,5})add(SpmvBackend::HybridDsmAtBsr,tile,0,{8,256,1,1024});
            }
            if(maximum>=4096)add(SpmvBackend::HybridDsmAtBsr,5,0,{8,128,1,4096});
            if(maximum>=128)for(int tile:{2,3}) {
                add(SpmvBackend::HybridDsmAtBsr,tile,0,{4,256,4,128});
                add(SpmvBackend::HybridDsmAtBsr,tile,0,{8,256,4,128});
            }
        }
    } else {
        if(o.backend==SpmvBackend::Auto||o.backend==SpmvBackend::AtBsr)
            for(int t=0;t<6;++t)for(int bi=0;bi<2;++bi) {
                int b=o.share_atbsr_tuning_storage?1-bi:bi;
                if((o.tile_index<0||t==o.tile_index)&&(o.balanced<0||b==o.balanced))add(SpmvBackend::AtBsr,t,b,{});
            }
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
    // A bounded measured refinement protects against vector-width prediction
    // errors while retaining the predicted candidate and default AtBSR.
    const bool refine_model=o.model_refine&&try_model;
    if(refine_model) {
        for(int width:{1,2,4,8,16,32}) {
            const auto& predicted=candidates[winner]->description;
            if(predicted.backend==SpmvBackend::CsrVector&&predicted.vector_width==width)continue;
            add(SpmvBackend::CsrVector,-1,-1,{},width);
        }
        s.info.model_refined=true;
    }
    const bool include_mixed=input.nnz&&((o.tune&&((o.backend==SpmvBackend::Auto&&o.enable_mixed_tile)||
                                                    o.backend==SpmvBackend::MixedTile))||
                                          (refine_model&&o.enable_mixed_tile));
    if(include_mixed) {
        try { for(int cap:{-1,0,4,8,16})add(SpmvBackend::MixedTile,-1,-1,{},0,cap); }
        catch(const CudaError& e) {
            if(o.backend!=SpmvBackend::Auto||e.code()!=cudaErrorMemoryAllocation)throw;
            mixed_packing.reset();cudaGetLastError();
            s.info.skipped_candidates.push_back("mixed_tile: insufficient GPU memory");
        } catch(const std::overflow_error& e) {
            if(o.backend!=SpmvBackend::Auto)throw;
            mixed_packing.reset();
            s.info.skipped_candidates.push_back(std::string("mixed_tile: ")+e.what());
        }
    }
    // Packed candidates own their row permutations and data; temporary CSR
    // and sort buffers must not remain resident during candidate timing.
    short_packing_input.reset();sorted_lengths.reset();sorted_indices.reset();
    if(candidates.empty())throw std::invalid_argument("no supported tuning candidates");
    if((o.tune||refine_model)&&input.nnz) {
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
        std::vector<bool> eligible;
        for(auto& candidate:candidates) {
            auto& c=*candidate;
            std::sort(c.samples.begin(),c.samples.end());
            c.description.median_us=(c.samples[(c.samples.size()-1)/2]+c.samples[c.samples.size()/2])/2;
            s.info.timings.push_back(c.description);
            const bool dsm_hybrid=c.description.backend==SpmvBackend::HybridDsmAtBsr||
                                  c.description.backend==SpmvBackend::CsrDsmVector;
            eligible.push_back(!dsm_hybrid||c.partition->nl>0);
        }
        winner=detail::selectMeasuredCandidate(s.info.timings,eligible,o.backend==SpmvBackend::Auto,
                                               o.hybrid_min_time_reduction,winner);
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
    hybrid_packing.clear();balanced_atbsr_packing.clear();partitions.clear();mixed_packing.reset();
    const auto& chosen=s.chosen->description;
    s.info.backend=chosen.backend;s.info.tile_index=chosen.tile_index;s.info.balanced=chosen.balanced;s.info.long_rows=chosen.long_rows;
    s.info.vector_width=chosen.vector_width;s.info.mixed_tile_cap=chosen.mixed_tile_cap;
    if(s.info.backend==SpmvBackend::AtBsr||s.info.backend==SpmvBackend::MixedTile) {s.ptr.reset();s.col.reset();s.val.reset();s.csr.row_ptr=nullptr;s.csr.col_idx=nullptr;s.csr.val=nullptr;}
    s.info.setup_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count();
    return plan;
    } catch(const CudaError& error) {
        if(error.code()!=cudaErrorMemoryAllocation||!can_retry())throw;
        return retry_with_shared();
    } catch(const SparseError& error) {
        if(error.code!=CUSPARSE_STATUS_ALLOC_FAILED||!can_retry())throw;
        return retry_with_shared();
    } catch(const std::bad_alloc& error) {
        // Thrust's private exception namespace varies with CUDA architecture
        // flags between translation units. Only its CUDA OOM diagnostic is
        // eligible; generic host std::bad_alloc must propagate unchanged.
        if(!can_retry()||std::string(error.what()).find(cudaGetErrorName(cudaErrorMemoryAllocation))==std::string::npos||
           std::string(error.what()).find(cudaGetErrorString(cudaErrorMemoryAllocation))==std::string::npos)throw;
        return retry_with_shared();
    }
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
