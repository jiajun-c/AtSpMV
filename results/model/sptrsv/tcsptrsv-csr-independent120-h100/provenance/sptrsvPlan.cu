#include "sptrsvPlan.h"
#include "detail/sptrsvPacking.h"
#include "detail/sptrsvShared.cuh"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace atspmv {
namespace {
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void check(cusparseStatus_t e) { if(e!=CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cuSPARSE error: "+std::to_string(int(e))); }
template<class T> void upload(T*& dst,const std::vector<T>& src,cudaStream_t stream) {
    check(cudaMalloc(&dst,std::max<std::size_t>(1,src.size())*sizeof(T)));
    if(!src.empty())check(cudaMemcpyAsync(dst,src.data(),src.size()*sizeof(T),cudaMemcpyHostToDevice,stream));
}
void validate(const SptrsvHostCsr& a,const double* b,double* x) {
    if(a.rows<=0 || !b || !x)throw std::invalid_argument("invalid SpTRSV dimensions or vector pointers");
    auto bi=reinterpret_cast<std::uintptr_t>(b),xi=reinterpret_cast<std::uintptr_t>(x);
    auto distance=bi>xi?bi-xi:xi-bi;
    if(distance<std::size_t(a.rows)*sizeof(double))throw std::invalid_argument("SpTRSV vectors overlap");
    if(a.values.size()!=a.col_idx.size() || a.values.size()>std::size_t(std::numeric_limits<int>::max()) ||
       a.row_ptr.size()!=std::size_t(a.rows)+1 || a.row_ptr.front()!=0 || a.row_ptr.back()!=int(a.values.size()))
        throw std::invalid_argument("invalid CSR array sizes");
    for(int i=0;i<a.rows;i++) {
        int start=a.row_ptr[i],end=a.row_ptr[i+1];
        if(start<0 || end<=start || end>int(a.values.size()))throw std::invalid_argument("invalid CSR row or missing diagonal");
        int prev=-1;
        for(int p=start;p<end;p++) {
            int col=a.col_idx[p];double value=a.values[p];
            if(col<=prev || col>i || !std::isfinite(value))throw std::invalid_argument("expected finite sorted unique lower-triangular CSR");
            prev=col;
        }
        if(a.col_idx[end-1]!=i || a.values[end-1]==0.)throw std::invalid_argument("missing or zero diagonal");
    }
}
}
struct SptrsvPlanState {
    SptrsvPlanInfo info;
    const double* rhs=nullptr;double* out=nullptr;
    int *ptr=nullptr,*col=nullptr,*tile_ptr=nullptr,*tile_col=nullptr,*tile_row_start=nullptr,*tile_rows=nullptr,*level_groups=nullptr,*widths=nullptr,*level_rows=nullptr,*level_offsets=nullptr;
    double *val=nullptr,*tile_val=nullptr;
    cusparseHandle_t handle=nullptr;cusparseSpMatDescr_t mat=nullptr;cusparseDnVecDescr_t b=nullptr,x=nullptr;
    cusparseSpSVDescr_t spsv=nullptr;void* buffer=nullptr;double alpha=1.;
    int ntiles=0,nslots=0,nrow_slots=0;
    ~SptrsvPlanState() {
        int previous=-1;cudaGetDevice(&previous);if(info.device>=0 && previous!=info.device)cudaSetDevice(info.device);
        if(spsv)cusparseSpSV_destroyDescr(spsv);if(b)cusparseDestroyDnVec(b);if(x)cusparseDestroyDnVec(x);
        if(mat)cusparseDestroySpMat(mat);if(handle)cusparseDestroy(handle);
        cudaFree(buffer);cudaFree(ptr);cudaFree(col);cudaFree(val);cudaFree(tile_ptr);cudaFree(tile_col);cudaFree(tile_val);
        cudaFree(tile_row_start);cudaFree(tile_rows);cudaFree(level_groups);cudaFree(widths);cudaFree(level_rows);cudaFree(level_offsets);
        if(previous>=0 && previous!=info.device)cudaSetDevice(previous);
    }
};
SptrsvPlan::SptrsvPlan()=default;SptrsvPlan::~SptrsvPlan()=default;
SptrsvPlan::SptrsvPlan(SptrsvPlan&&) noexcept=default;
SptrsvPlan& SptrsvPlan::operator=(SptrsvPlan&&) noexcept=default;
const SptrsvPlanInfo& SptrsvPlan::info() const {if(!state_)throw std::logic_error("empty SpTRSV plan");return state_->info;}
SptrsvPlan makeSptrsvPlan(const SptrsvHostCsr& input,const double* rhs,double* out,const SptrsvPlanOptions& options,cudaStream_t stream) {
    validate(input,rhs,out);
    SptrsvPlan plan;plan.state_=std::make_unique<SptrsvPlanState>();auto& s=*plan.state_;s.rhs=rhs;s.out=out;
    check(cudaGetDevice(&s.info.device));cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,s.info.device));int limit=0;
    check(cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,s.info.device));
    sptrsv_detail::Matrix m{input.rows,input.row_ptr,input.col_idx,input.values};
    sptrsv_detail::Levels levels;sptrsv_detail::Packed packed;std::vector<int> starts;
    const bool calibrated=prop.major==9 && prop.multiProcessorCount==114 && std::string(prop.name).find("H100 PCIe")!=std::string::npos;
    s.info.rows=m.n;s.info.nnz=m.val.size();s.info.levels=-1;
    s.info.backend=SptrsvBackend::CuSparse;
    if(options.enable_cached_tile_model && calibrated && m.n<=4000) {
        levels=sptrsv_detail::make_levels(m);s.info.levels=levels.width.size();
        std::size_t groups=0,row_slots=0,slots=0;
        for(std::size_t l=0;l<levels.width.size();l++) {
            int k=levels.width[l],height=32/k;
            for(int first=levels.offsets[l];first<levels.offsets[l+1];first+=height) {
                int maxlen=0;
                for(int r=first;r<std::min(first+height,levels.offsets[l+1]);r++) {
                    int row=levels.rows[r];maxlen=std::max(maxlen,m.ptr[row+1]-m.ptr[row]-1);
                }
                groups++;row_slots+=height;slots+=std::size_t((maxlen+k-1)/k)*32;
            }
        }
        std::size_t bytes=sizeof(double)*(std::size_t(3)*m.n+slots)+sizeof(int)*(groups+1+slots+groups+row_slots+levels.width.size()+1+levels.width.size());
        std::size_t csr_bytes=sizeof(double)*(std::size_t(2)*m.n+m.val.size())+
            sizeof(int)*(m.ptr.size()+m.col.size()+levels.rows.size()+levels.offsets.size());
        s.info.backend=selectSptrsvBackend({m.n,int(levels.width.size()),bytes,std::size_t(limit),true,csr_bytes});
        s.info.cached_bytes=s.info.backend==SptrsvBackend::CachedCsr?csr_bytes:bytes;
        if(s.info.backend==SptrsvBackend::CachedTile) {
            s.info.cached_threads=selectSptrsvTileThreads(m.n,s.info.levels);
            packed=sptrsv_detail::pack_levels(m,levels);int row_base=0;
            for(std::size_t l=0;l<levels.width.size();l++)for(int g=packed.group_offsets[l];g<packed.group_offsets[l+1];g++){starts.push_back(row_base);row_base+=32/levels.width[l];}
        }
    }
    upload(s.ptr,m.ptr,stream);upload(s.val,m.val,stream);
    if(s.info.backend==SptrsvBackend::CachedTile) {
        s.ntiles=starts.size();s.nslots=packed.val.size();s.nrow_slots=packed.rows.size();
        upload(s.tile_ptr,packed.ptr,stream);upload(s.tile_col,packed.col,stream);upload(s.tile_val,packed.val,stream);
        upload(s.tile_rows,packed.rows,stream);upload(s.tile_row_start,starts,stream);upload(s.level_groups,packed.group_offsets,stream);upload(s.widths,levels.width,stream);
        // This attribute is shared by all plans using this kernel. Never lower it
        // to an individual plan size: other live plans may need more.
        check(cudaFuncSetAttribute(solve_cached_tiles,cudaFuncAttributeMaxDynamicSharedMemorySize,limit));
        check(cudaStreamSynchronize(stream));
    } else if(s.info.backend==SptrsvBackend::CachedCsr) {
        s.info.cached_threads=256;
        upload(s.col,m.col,stream);upload(s.level_rows,levels.rows,stream);upload(s.level_offsets,levels.offsets,stream);
        check(cudaFuncSetAttribute(solve_cached_csr<32>,cudaFuncAttributeMaxDynamicSharedMemorySize,limit));
        check(cudaStreamSynchronize(stream));
    } else {
        upload(s.col,m.col,stream);check(cudaStreamSynchronize(stream));
        check(cusparseCreate(&s.handle));check(cusparseSetStream(s.handle,stream));
        check(cusparseCreateCsr(&s.mat,m.n,m.n,m.val.size(),s.ptr,s.col,s.val,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
        cusparseFillMode_t fill=CUSPARSE_FILL_MODE_LOWER;cusparseDiagType_t diag=CUSPARSE_DIAG_TYPE_NON_UNIT;
        check(cusparseSpMatSetAttribute(s.mat,CUSPARSE_SPMAT_FILL_MODE,&fill,sizeof(fill)));check(cusparseSpMatSetAttribute(s.mat,CUSPARSE_SPMAT_DIAG_TYPE,&diag,sizeof(diag)));
        check(cusparseCreateDnVec(&s.b,m.n,const_cast<double*>(rhs),CUDA_R_64F));check(cusparseCreateDnVec(&s.x,m.n,out,CUDA_R_64F));check(cusparseSpSV_createDescr(&s.spsv));
        std::size_t size=0;check(cusparseSpSV_bufferSize(s.handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&s.alpha,s.mat,s.b,s.x,CUDA_R_64F,CUSPARSE_SPSV_ALG_DEFAULT,s.spsv,&size));
        check(cudaMalloc(&s.buffer,std::max<std::size_t>(1,size)));
        check(cusparseSpSV_analysis(s.handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&s.alpha,s.mat,s.b,s.x,CUDA_R_64F,CUSPARSE_SPSV_ALG_DEFAULT,s.spsv,s.buffer));check(cudaStreamSynchronize(stream));
    }
    return plan;
}
void executeSptrsv(const SptrsvPlan& plan,cudaStream_t stream) {
    if(!plan.state_)throw std::logic_error("empty SpTRSV plan");auto& s=*plan.state_;int device;check(cudaGetDevice(&device));
    if(device!=s.info.device)throw std::invalid_argument("SpTRSV plan device mismatch");
    if(s.info.backend==SptrsvBackend::CachedTile) {
        solve_cached_tiles<<<1,s.info.cached_threads,s.info.cached_bytes,stream>>>(s.tile_ptr,s.tile_col,s.tile_val,s.tile_row_start,s.tile_rows,s.level_groups,s.widths,s.ptr,s.val,s.info.levels,s.info.rows,s.ntiles,s.nslots,s.nrow_slots,s.rhs,s.out);check(cudaGetLastError());
    } else if(s.info.backend==SptrsvBackend::CachedCsr) {
        solve_cached_csr<32><<<1,s.info.cached_threads,s.info.cached_bytes,stream>>>(s.ptr,s.col,s.val,s.level_rows,s.level_offsets,s.info.levels,s.info.rows,s.info.nnz,s.rhs,s.out);check(cudaGetLastError());
    } else {check(cusparseSetStream(s.handle,stream));check(cusparseSpSV_solve(s.handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&s.alpha,s.mat,s.b,s.x,CUDA_R_64F,CUSPARSE_SPSV_ALG_DEFAULT,s.spsv));}
}
} // namespace atspmv
