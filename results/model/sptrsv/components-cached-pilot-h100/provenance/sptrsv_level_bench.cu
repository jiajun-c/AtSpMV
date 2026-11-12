// Experimental level-scheduled SpTRSV baseline; see docs/sptrsv-experiment.md.
#include "csrReader.h"
#include "sptrsvSelection.h"
#include "detail/sptrsvPacking.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA(call) do { cudaError_t cuda_status_=(call); if(cuda_status_!=cudaSuccess) throw std::runtime_error(std::string(#call)+": "+cudaGetErrorString(cuda_status_)); } while(0)
#define SP(call) do { cusparseStatus_t e=(call); if(e!=CUSPARSE_STATUS_SUCCESS) throw std::runtime_error(std::string(#call)+": status "+std::to_string(e)); } while(0)
using Clock=std::chrono::steady_clock;
using VecI=std::vector<int>;
using VecD=std::vector<double>;
using atspmv::sptrsv_detail::Matrix;
using atspmv::sptrsv_detail::Levels;
using atspmv::sptrsv_detail::Packed;
using atspmv::sptrsv_detail::make_levels;
using atspmv::sptrsv_detail::pack_levels;
template<class T> T* upload(const std::vector<T>& v) {
    T* p=nullptr; CUDA(cudaMalloc(&p,std::max<size_t>(1,v.size())*sizeof(T)));
    if(!v.empty()) CUDA(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); return p;
}
// Extract the strict lower pattern. The input values are intentionally replaced
// with bounded, diagonally dominant ones to make residual checks meaningful.
Matrix make_matrix(const std::string& input, bool factor) {
    Matrix a; VecI inptr,incol;
    if(factor) {
        if(input=="--synthetic") throw std::runtime_error("--factor requires a Matrix Market file");
        auto src=read_mtx(input);
        if(src.nrows!=src.ncols || src.nrows<=0) throw std::runtime_error("input must be nonempty and square");
        a.n=src.nrows; a.ptr.push_back(0);
        for(int row=0;row<a.n;row++) {
            std::vector<std::pair<int,double>> entries;
            for(int p=src.row_ptr[row];p<src.row_ptr[row+1];p++) {
                int j=src.col_idx[p]; double v=src.val[p];
                if(j<0 || j>=a.n || j>row || !std::isfinite(v)) throw std::runtime_error("factor must contain finite lower triangular entries only");
                entries.emplace_back(j,v);
            }
            std::sort(entries.begin(),entries.end(),[](auto x,auto y){return x.first<y.first;});
            for(size_t i=0;i<entries.size();) {
                int j=entries[i].first; double v=0.;
                while(i<entries.size() && entries[i].first==j) v+=entries[i++].second;
                if(!std::isfinite(v)) throw std::runtime_error("nonfinite merged entry");
                if(v!=0.) {a.col.push_back(j);a.val.push_back(v);}
            }
            if(a.col.size()==size_t(a.ptr.back()) || a.col.back()!=row) throw std::runtime_error("factor has missing or zero diagonal");
            a.ptr.push_back(a.col.size());
        }
        delete[] src.row_ptr; delete[] src.col_idx; delete[] src.val;
        return a;
    }
    if(input=="--synthetic") {
        a.n=16384; inptr.push_back(0);
        for(int i=0;i<a.n;i++) { // 16 layers, 1024 independent rows per layer
            int layer=i/1024; for(int k=0;k<24 && layer;k++) incol.push_back((layer-1)*1024+(i*13+k*41)%1024);
            inptr.push_back(incol.size());
        }
    } else {
        auto src=read_mtx(input); if(src.nrows!=src.ncols || src.nrows<=0) throw std::runtime_error("input must be nonempty and square");
        a.n=src.nrows; inptr.assign(src.row_ptr,src.row_ptr+a.n+1); incol.assign(src.col_idx,src.col_idx+src.nnz);
        delete[] src.row_ptr; delete[] src.col_idx; delete[] src.val;
    }
    a.ptr.push_back(0);
    for(int i=0;i<a.n;i++) {
        VecI cols; for(int p=inptr[i];p<inptr[i+1];p++) if(incol[p]>=0 && incol[p]<i) cols.push_back(incol[p]);
        std::sort(cols.begin(),cols.end()); cols.erase(std::unique(cols.begin(),cols.end()),cols.end());
        for(int j:cols) {a.col.push_back(j); a.val.push_back(-0.5/std::max<size_t>(1,cols.size()));}
        a.col.push_back(i); a.val.push_back(1.0); a.ptr.push_back(a.col.size());
    }
    return a;
}
template<int K> __global__ void packed_level(const int* ptr,const int* col,const double* val,const int* packed_rows,
    int group_start,int n_groups,int row_start,const int* csr_ptr,const double* csr_val,const double* b,double* x) {
    constexpr int M=32/K; int warp=threadIdx.x/32, lane=threadIdx.x%32;
    int local=blockIdx.x*(blockDim.x/32)+warp; if(local>=n_groups) return;
    int group=group_start+local, row=packed_rows[row_start+local*M+lane/K]; double sum=0;
    for(int p=ptr[group]+lane;p<ptr[group+1];p+=32) {int j=col[p]; if(j>=0) sum+=val[p]*x[j];}
    #pragma unroll
    for(int d=K/2;d;d/=2) sum+=__shfl_down_sync(0xffffffff,sum,d,K);
    if(lane%K==0 && row>=0) x[row]=(b[row]-sum)/csr_val[csr_ptr[row+1]-1];
}
void launch_packed(int k,int groups,int start,int row_start,const int* pp,const int* pc,const double* pv,const int* pr,const int* cp,const double* cv,const double* b,double* x,cudaStream_t stream) {
    int blocks=(groups+3)/4;
    switch(k) {
#define PACKED_CASE(K) case K: packed_level<K><<<blocks,128,0,stream>>>(pp,pc,pv,pr,start,groups,row_start,cp,cv,b,x); break
    PACKED_CASE(1); PACKED_CASE(4); PACKED_CASE(8); PACKED_CASE(16); PACKED_CASE(32);
#undef PACKED_CASE
    default: throw std::runtime_error("bad packed width"); }
    CUDA(cudaGetLastError());
}
// A single resident CTA processes all levels and uses block barriers. This is
// a useful small/narrow-level candidate and avoids global spinning/deadlock.
template<int W> __global__ void solve_single_cta(const int* ptr,const int* col,const double* val,const int* rows,const int* offsets,int nlevels,const double* b,double* x) {
    int group=threadIdx.x/W, lane=threadIdx.x%W, groups=blockDim.x/W;
    for(int level=0;level<nlevels;level++) {
        int count=offsets[level+1]-offsets[level];
        for(int item=group;item<count;item+=groups) {
            int row=rows[offsets[level]+item]; double sum=0;
            for(int p=ptr[row]+lane;p<ptr[row+1]-1;p+=W) sum+=val[p]*x[col[p]];
            unsigned active=__activemask();
            #pragma unroll
            for(int d=W/2;d;d/=2) sum+=__shfl_down_sync(active,sum,d,W);
            if(lane==0) x[row]=(b[row]-sum)/val[ptr[row+1]-1];
        }
        __syncthreads();
    }
}
template<int W> __global__ void solve_level(const int* ptr,const int* col,const double* val,const int* rows,int start,int count,const double* b,double* x) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x, group=tid/W, lane=threadIdx.x%W;
    if(group>=count) return;
    int row=rows[start+group]; double sum=0;
    for(int p=ptr[row]+lane;p<ptr[row+1]-1;p+=W) sum+=val[p]*x[col[p]];
    unsigned active=__activemask();
    #pragma unroll
    for(int offset=W/2;offset;offset/=2) sum+=__shfl_down_sync(active,sum,offset,W);
    if(lane==0) x[row]=(b[row]-sum)/val[ptr[row+1]-1];
}
void launch(int width,int count,int start,const int* p,const int* c,const double* v,const int* r,const double* b,double* x,cudaStream_t stream) {
    int blocks=(count*width+127)/128;
    switch(width) {
#define CASE(W) case W: solve_level<W><<<blocks,128,0,stream>>>(p,c,v,r,start,count,b,x); break
    CASE(1); CASE(2); CASE(4); CASE(8); CASE(16); CASE(32);
#undef CASE
    default: throw std::runtime_error("bad width"); }
    CUDA(cudaGetLastError());
}
struct Graph {cudaGraph_t graph{}; cudaGraphExec_t exec{};};
template<class F> Graph capture(cudaStream_t stream,F f) {
    Graph g; CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal)); f(); CUDA(cudaStreamEndCapture(stream,&g.graph));
    CUDA(cudaGraphInstantiate(&g.exec,g.graph,nullptr,nullptr,0)); return g;
}
double measure(Graph g,cudaStream_t stream) {
    for(int i=0;i<5;i++) CUDA(cudaGraphLaunch(g.exec,stream)); CUDA(cudaStreamSynchronize(stream));
    cudaEvent_t s,e; CUDA(cudaEventCreate(&s)); CUDA(cudaEventCreate(&e)); std::vector<double> times;
    for(int k=0;k<9;k++) { CUDA(cudaEventRecord(s,stream)); for(int i=0;i<10;i++) CUDA(cudaGraphLaunch(g.exec,stream)); CUDA(cudaEventRecord(e,stream)); CUDA(cudaEventSynchronize(e)); float ms; CUDA(cudaEventElapsedTime(&ms,s,e)); times.push_back(ms*100.0); }
    CUDA(cudaEventDestroy(s)); CUDA(cudaEventDestroy(e)); std::sort(times.begin(),times.end()); return times[times.size()/2];
}
void destroy(Graph g) { CUDA(cudaGraphExecDestroy(g.exec)); CUDA(cudaGraphDestroy(g.graph)); }
#include "sptrsv_blocked.cuh"
#include "sptrsv_queue.cuh"
#include "sptrsv_ordered.cuh"
#include "sptrsv_push.cuh"
#include "sptrsv_tile_wait.cuh"
#include "detail/sptrsvShared.cuh"
#include "sptrsv_component_cached.cuh"
int main(int argc,char**argv) { try {
    if(argc!=2 && !(argc==3 && std::string(argv[2])=="--factor")) throw std::runtime_error("usage: sptrsv_level_bench --synthetic|matrix.mtx [--factor]");
    bool cached_only=std::getenv("ATSPTRSV_CACHED_ONLY")!=nullptr;
    if(std::getenv("ATSPTRSV_TILE_K2") && !cached_only)throw std::invalid_argument("ATSPTRSV_TILE_K2 requires ATSPTRSV_CACHED_ONLY");
    auto a=make_matrix(argv[1],argc==3); auto begin=Clock::now(); auto levels=make_levels(a,std::getenv("ATSPTRSV_TILE_K2")!=nullptr); auto end=Clock::now();
    if(levels.width.size()>4096) throw std::runtime_error("more than 4096 dependency levels: graph baseline intentionally capped");
    VecD truth(a.n),rhs(a.n,0),out(a.n); for(int i=0;i<a.n;i++) truth[i]=0.5+double((i*37LL)%101)/101;
    for(int i=0;i<a.n;i++) for(int p=a.ptr[i];p<a.ptr[i+1];p++) rhs[i]+=a.val[p]*truth[a.col[p]];
    int *dp=upload(a.ptr),*dc=upload(a.col),*dr=upload(levels.rows); double *dv=upload(a.val),*db=upload(rhs),*dx=upload(VecD(a.n,0));
    cudaStream_t stream; CUDA(cudaStreamCreate(&stream));
    cusparseHandle_t h; SP(cusparseCreate(&h)); SP(cusparseSetStream(h,stream));
    cusparseSpMatDescr_t mat; cusparseDnVecDescr_t b,x; cusparseSpSVDescr_t desc;
    SP(cusparseCreateCsr(&mat,a.n,a.n,a.val.size(),dp,dc,dv,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
    cusparseFillMode_t fill=CUSPARSE_FILL_MODE_LOWER; cusparseDiagType_t diag=CUSPARSE_DIAG_TYPE_NON_UNIT;
    SP(cusparseSpMatSetAttribute(mat,CUSPARSE_SPMAT_FILL_MODE,&fill,sizeof(fill))); SP(cusparseSpMatSetAttribute(mat,CUSPARSE_SPMAT_DIAG_TYPE,&diag,sizeof(diag)));
    SP(cusparseCreateDnVec(&b,a.n,db,CUDA_R_64F)); SP(cusparseCreateDnVec(&x,a.n,dx,CUDA_R_64F)); SP(cusparseSpSV_createDescr(&desc));
    double alpha=1; size_t bytes=0; SP(cusparseSpSV_bufferSize(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,b,x,CUDA_R_64F,CUSPARSE_SPSV_ALG_DEFAULT,desc,&bytes));
    void* buf; CUDA(cudaMalloc(&buf,std::max<size_t>(1,bytes))); CUDA(cudaStreamSynchronize(stream)); auto cs=Clock::now();
    SP(cusparseSpSV_analysis(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,b,x,CUDA_R_64F,CUSPARSE_SPSV_ALG_DEFAULT,desc,buf)); CUDA(cudaStreamSynchronize(stream)); auto ce=Clock::now();
    double backward=0;
    auto check_result=[&](const std::string& name){
        CUDA(cudaStreamSynchronize(stream)); CUDA(cudaMemcpy(out.data(),dx,a.n*sizeof(double),cudaMemcpyDeviceToHost));
        double err=0; long double max_r=0, norm_a=0, norm_x=0, norm_b=0;
        for(int i=0;i<a.n;i++) {
            if(!std::isfinite(out[i])) throw std::runtime_error(name+" produced nonfinite output");
            err=std::max(err,std::abs(out[i]-truth[i])); norm_x=std::max(norm_x,std::abs((long double)out[i])); norm_b=std::max(norm_b,std::abs((long double)rhs[i]));
            long double row=0, magnitude=0; for(int p=a.ptr[i];p<a.ptr[i+1];p++) {row+=(long double)a.val[p]*out[a.col[p]]; magnitude+=std::abs((long double)a.val[p]);}
            max_r=std::max(max_r,std::abs(row-(long double)rhs[i])); norm_a=std::max(norm_a,magnitude);
        }
        long double denominator=norm_a*norm_x+norm_b; backward=denominator?double(max_r/denominator):double(max_r);
        if(!std::isfinite(backward) || backward>1e-12 || (argc==2 && err>1e-10)) throw std::runtime_error(name+" incorrect: backward error "+std::to_string(backward)+", forward error "+std::to_string(err));
        return err;
    };
    // Replaying with a different right-hand side prevents a stale solution
    // from passing the correctness check even if the solve were skipped.
    VecD rhs_alt(a.n,0),truth_alt(a.n);
    for(int i=0;i<a.n;i++) truth_alt[i]=-0.75+double((i*19LL)%73)/71.;
    for(int i=0;i<a.n;i++) for(int p=a.ptr[i];p<a.ptr[i+1];p++) rhs_alt[i]+=a.val[p]*truth_alt[a.col[p]];
    auto check_alternate=[&](Graph g) {
        double saved_backward=backward;
        std::swap(rhs,rhs_alt); std::swap(truth,truth_alt);
        CUDA(cudaMemcpyAsync(db,rhs.data(),a.n*sizeof(double),cudaMemcpyHostToDevice,stream));
        CUDA(cudaMemsetAsync(dx,0,a.n*sizeof(double),stream));CUDA(cudaGraphLaunch(g.exec,stream));check_result("ordered alternate rhs");
        std::swap(rhs,rhs_alt);std::swap(truth,truth_alt);
        CUDA(cudaMemcpyAsync(db,rhs.data(),a.n*sizeof(double),cudaMemcpyHostToDevice,stream));
        CUDA(cudaStreamSynchronize(stream));backward=saved_backward;
    };
    auto cg=capture(stream,[&]{SP(cusparseSpSV_solve(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,b,x,CUDA_R_64F,CUSPARSE_SPSV_ALG_DEFAULT,desc));});
    CUDA(cudaMemsetAsync(dx,0,a.n*sizeof(double),stream)); CUDA(cudaGraphLaunch(cg.exec,stream)); double error=check_result("cusparse"); double us=measure(cg,stream); destroy(cg);
    std::cout<<std::setprecision(10)<<"matrix,n,nnz,levels,variant,analysis_us,solve_us,max_error,backward_error\n";
    std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<",cusparse,"<<std::chrono::duration<double,std::micro>(ce-cs).count()<<','<<us<<','<<error<<','<<backward<<'\n';
    for(int i=0;i<5;i++) {SP(cusparseSpSV_solve(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,b,x,CUDA_R_64F,CUSPARSE_SPSV_ALG_DEFAULT,desc));CUDA(cudaDeviceSynchronize());}
    auto host_start=Clock::now();
    for(int i=0;i<200;i++) {SP(cusparseSpSV_solve(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,b,x,CUDA_R_64F,CUSPARSE_SPSV_ALG_DEFAULT,desc));CUDA(cudaDeviceSynchronize());}
    auto host_end=Clock::now(); error=check_result("cusparse host");
    std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<",cusparse-host,"<<std::chrono::duration<double,std::micro>(ce-cs).count()<<','<<std::chrono::duration<double,std::micro>(host_end-host_start).count()/200<<','<<error<<','<<backward<<'\n';
    auto pack_start=Clock::now(); auto packed=pack_levels(a,levels); auto pack_end=Clock::now();
    int *pp=upload(packed.ptr),*pc=upload(packed.col),*pr=upload(packed.rows); double* pv=upload(packed.val);
    if(!cached_only) {
    auto pg=capture(stream,[&]{int row_start=0; for(size_t s=0;s<levels.width.size();s++) {
        int k=levels.width[s], start=packed.group_offsets[s], groups=packed.group_offsets[s+1]-start;
        launch_packed(k,groups,start,row_start,pp,pc,pv,pr,dp,dv,db,dx,stream); row_start+=groups*(32/k);
    }});
    CUDA(cudaMemsetAsync(dx,0,a.n*sizeof(double),stream)); CUDA(cudaGraphLaunch(pg.exec,stream)); error=check_result("packed adaptive"); us=measure(pg,stream); destroy(pg);
    std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<",packed-adaptive,"<<(std::chrono::duration<double,std::micro>(end-begin).count()+std::chrono::duration<double,std::micro>(pack_end-pack_start).count())<<','<<us<<','<<error<<','<<backward<<'\n';
    std::cerr<<"packed_slots="<<packed.val.size()<<",strict_lower_nnz="<<(a.val.size()-a.n)<<",packing_us="<<std::chrono::duration<double,std::micro>(pack_end-pack_start).count()<<'\n';
    }
    VecI task_width,task_row_start;int row_base=0;
    for(size_t level=0;level<levels.width.size();++level) {
        int k=levels.width[level],groups=packed.group_offsets[level+1]-packed.group_offsets[level];
        for(int g=0;g<groups;g++) {task_width.push_back(k);task_row_start.push_back(row_base);row_base+=32/k;}
    }
    int *tw=upload(task_width),*tr=upload(task_row_start),*tn,*tready;int ntiles=task_width.size();
    CUDA(cudaMalloc(&tn,sizeof(int)));CUDA(cudaMalloc(&tready,a.n*sizeof(int)));
    int tile_device,tile_sm;CUDA(cudaGetDevice(&tile_device));CUDA(cudaDeviceGetAttribute(&tile_sm,cudaDevAttrMultiProcessorCount,tile_device));
    if(!cached_only) for(int grid: {16,tile_sm}) {
        auto tg=capture(stream,[&]{
            CUDA(cudaMemsetAsync(tn,0,sizeof(int),stream));CUDA(cudaMemsetAsync(tready,0,a.n*sizeof(int),stream));
            void* args[]={&pp,&pc,&pv,&tr,&tw,&pr,&ntiles,&tn,&tready,&dp,&dv,&db,&dx};
            CUDA(cudaLaunchCooperativeKernel((const void*)tile_wait_solve,dim3(grid),dim3(128),args,0,stream));
        });
        CUDA(cudaMemsetAsync(dx,0,a.n*sizeof(double),stream));CUDA(cudaGraphLaunch(tg.exec,stream));error=check_result("tile wait");
        us=measure(tg,stream);error=std::max(error,check_result("tile wait replay"));check_alternate(tg);destroy(tg);
        std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<",tile-wait-"<<grid<<','<<(std::chrono::duration<double,std::micro>(end-begin).count()+std::chrono::duration<double,std::micro>(pack_end-pack_start).count())<<','<<us<<','<<error<<','<<backward<<'\n';
    }
    int shared_limit=0;CUDA(cudaDeviceGetAttribute(&shared_limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,tile_device));
    size_t shared_bytes=size_t(a.n)*sizeof(double);
    if(shared_bytes<=size_t(shared_limit)) {
        int* levelgroups=upload(packed.group_offsets);int* widths=upload(levels.width);
        CUDA(cudaFuncSetAttribute(solve_shared_tiles,cudaFuncAttributeMaxDynamicSharedMemorySize,shared_bytes));
        if(!cached_only) for(int threads: {128,256,512}) {
            auto sg=capture(stream,[&]{solve_shared_tiles<<<1,threads,shared_bytes,stream>>>(pp,pc,pv,tr,pr,levelgroups,widths,dp,dv,levels.width.size(),a.n,db,dx);CUDA(cudaGetLastError());});
            CUDA(cudaMemsetAsync(dx,0,a.n*sizeof(double),stream));CUDA(cudaGraphLaunch(sg.exec,stream));error=check_result("shared tiles");
            us=measure(sg,stream);error=std::max(error,check_result("shared tiles replay"));check_alternate(sg);destroy(sg);
            std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<",shared-tiles-"<<threads<<','<<(std::chrono::duration<double,std::micro>(end-begin).count()+std::chrono::duration<double,std::micro>(pack_end-pack_start).count())<<','<<us<<','<<error<<','<<backward<<'\n';
        }
        size_t bytes=sizeof(double)*(size_t(3)*a.n+packed.val.size())+sizeof(int)*(packed.ptr.size()+packed.col.size()+task_row_start.size()+packed.rows.size()+packed.group_offsets.size()+levels.width.size());
        if(bytes<=size_t(shared_limit)) {
            CUDA(cudaFuncSetAttribute(solve_cached_tiles,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
            std::vector<int> thread_counts{128,256,512};
            if(std::getenv("ATSPTRSV_THREADS_512_1024"))thread_counts={512,1024};
            if(std::getenv("ATSPTRSV_REVERSE_THREADS"))std::reverse(thread_counts.begin(),thread_counts.end());
            for(int threads: thread_counts) {
                auto cg=capture(stream,[&]{solve_cached_tiles<<<1,threads,bytes,stream>>>(pp,pc,pv,tr,pr,levelgroups,widths,dp,dv,levels.width.size(),a.n,ntiles,packed.val.size(),packed.rows.size(),db,dx);CUDA(cudaGetLastError());});
                CUDA(cudaMemsetAsync(dx,0,a.n*sizeof(double),stream));CUDA(cudaGraphLaunch(cg.exec,stream));error=check_result("cached tiles");us=measure(cg,stream);
                error=std::max(error,check_result("cached tiles replay"));check_alternate(cg);destroy(cg);
                std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<",cached-tiles-"<<threads<<','<<(std::chrono::duration<double,std::micro>(end-begin).count()+std::chrono::duration<double,std::micro>(pack_end-pack_start).count())<<','<<us<<','<<error<<','<<backward<<'\n';
                auto run=[&]{solve_cached_tiles<<<1,threads,bytes,stream>>>(pp,pc,pv,tr,pr,levelgroups,widths,dp,dv,levels.width.size(),a.n,ntiles,packed.val.size(),packed.rows.size(),db,dx);CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());};
                for(int i=0;i<5;i++)run();auto hstart=Clock::now();for(int i=0;i<200;i++)run();auto hend=Clock::now();error=check_result("cached tiles host");
                std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<",cached-tiles-host-"<<threads<<','<<(std::chrono::duration<double,std::micro>(end-begin).count()+std::chrono::duration<double,std::micro>(pack_end-pack_start).count())<<','<<std::chrono::duration<double,std::micro>(hend-hstart).count()/200<<','<<error<<','<<backward<<'\n';
            }
        }
        if(cached_only) {
            cudaDeviceProp prop{};CUDA(cudaGetDeviceProperties(&prop,tile_device));
            atspmv::SptrsvSelectionFeatures features{a.n,int(levels.width.size()),bytes,size_t(shared_limit),
                prop.major==9 && prop.multiProcessorCount==114 && std::string(prop.name).find("H100 PCIe")!=std::string::npos};
            auto selected=atspmv::selectSptrsvBackend(features);
            auto selected_graph=capture(stream,[&]{
                if(selected==atspmv::SptrsvBackend::CachedTile) {
                    solve_cached_tiles<<<1,512,bytes,stream>>>(pp,pc,pv,tr,pr,levelgroups,widths,dp,dv,levels.width.size(),a.n,ntiles,packed.val.size(),packed.rows.size(),db,dx);CUDA(cudaGetLastError());
                } else SP(cusparseSpSV_solve(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,b,x,CUDA_R_64F,CUSPARSE_SPSV_ALG_DEFAULT,desc));
            });
            CUDA(cudaMemsetAsync(dx,0,a.n*sizeof(double),stream));CUDA(cudaGraphLaunch(selected_graph.exec,stream));error=check_result("selector");
            us=measure(selected_graph,stream);error=std::max(error,check_result("selector replay"));check_alternate(selected_graph);destroy(selected_graph);
            std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<','<<(selected==atspmv::SptrsvBackend::CachedTile?"selector-cached":"selector-cusparse")<<','<<(std::chrono::duration<double,std::micro>(end-begin).count()+std::chrono::duration<double,std::micro>(pack_end-pack_start).count())<<','<<us<<','<<error<<','<<backward<<'\n';
        }
        CUDA(cudaFree(levelgroups));CUDA(cudaFree(widths));
    }
    if(cached_only && shared_bytes>size_t(shared_limit)) {
        auto sg=capture(stream,[&]{SP(cusparseSpSV_solve(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,b,x,CUDA_R_64F,CUSPARSE_SPSV_ALG_DEFAULT,desc));});
        CUDA(cudaMemsetAsync(dx,0,a.n*sizeof(double),stream));CUDA(cudaGraphLaunch(sg.exec,stream));error=check_result("selector fallback");us=measure(sg,stream);
        error=std::max(error,check_result("selector fallback replay"));check_alternate(sg);destroy(sg);
        std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<",selector-cusparse,"<<std::chrono::duration<double,std::micro>(end-begin).count()<<','<<us<<','<<error<<','<<backward<<'\n';
    }
    CUDA(cudaFree(tw));CUDA(cudaFree(tr));CUDA(cudaFree(tn));CUDA(cudaFree(tready));
    CUDA(cudaFree(pp)); CUDA(cudaFree(pc)); CUDA(cudaFree(pr)); CUDA(cudaFree(pv));
    if(!cached_only) {
    auto component_start=Clock::now();auto ccached=pack_cached_components(a);auto component_end=Clock::now();
    if(ccached.max_shared_bytes<=size_t(shared_limit)) {
        auto *cm=upload(ccached.meta);int *map=upload(ccached.map),*cp=upload(ccached.ptr),*cidx=upload(ccached.col);
        int *cgs=upload(ccached.group_start),*cr=upload(ccached.rows),*cl=upload(ccached.level_group),*cw=upload(ccached.width);
        double *cv=upload(ccached.val),*diag=upload(ccached.diagonal);
        CUDA(cudaFuncSetAttribute(solve_cached_components,cudaFuncAttributeMaxDynamicSharedMemorySize,shared_limit));
        for(int threads: {256,512}) {
            auto cg=capture(stream,[&]{solve_cached_components<<<ccached.meta.size(),threads,ccached.max_shared_bytes,stream>>>(cm,map,cp,cidx,cv,diag,cgs,cr,cl,cw,db,dx);CUDA(cudaGetLastError());});
            CUDA(cudaMemsetAsync(dx,0,a.n*sizeof(double),stream));CUDA(cudaGraphLaunch(cg.exec,stream));error=check_result("cached components");
            us=measure(cg,stream);error=std::max(error,check_result("cached components replay"));check_alternate(cg);destroy(cg);
            std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<",cached-components-"<<threads<<','<<std::chrono::duration<double,std::micro>(component_end-component_start).count()<<','<<us<<','<<error<<','<<backward<<'\n';
        }
        std::cerr<<"cached_components="<<ccached.meta.size()<<",max_shared_bytes="<<ccached.max_shared_bytes<<'\n';
        CUDA(cudaFree(cm));CUDA(cudaFree(map));CUDA(cudaFree(cp));CUDA(cudaFree(cidx));CUDA(cudaFree(cgs));CUDA(cudaFree(cr));CUDA(cudaFree(cl));CUDA(cudaFree(cw));CUDA(cudaFree(cv));CUDA(cudaFree(diag));
    } else std::cerr<<"cached_components_skipped_shared_bytes="<<ccached.max_shared_bytes<<'\n';
    auto sorted_start=Clock::now(); auto sorted=pack_levels(a,levels,true); auto sorted_end=Clock::now();
    int *spp=upload(sorted.ptr),*spc=upload(sorted.col),*spr=upload(sorted.rows); double* spv=upload(sorted.val);
    auto sg=capture(stream,[&]{int row_start=0; for(size_t s=0;s<levels.width.size();s++) {
        int k=levels.width[s], start=sorted.group_offsets[s], groups=sorted.group_offsets[s+1]-start;
        launch_packed(k,groups,start,row_start,spp,spc,spv,spr,dp,dv,db,dx,stream); row_start+=groups*(32/k);
    }});
    CUDA(cudaMemsetAsync(dx,0,a.n*sizeof(double),stream)); CUDA(cudaGraphLaunch(sg.exec,stream)); error=check_result("sorted packed"); us=measure(sg,stream); destroy(sg);
    std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<",packed-sorted,"<<(std::chrono::duration<double,std::micro>(end-begin).count()+std::chrono::duration<double,std::micro>(sorted_end-sorted_start).count())<<','<<us<<','<<error<<','<<backward<<'\n';
    std::cerr<<"sorted_slots="<<sorted.val.size()<<'\n';
    CUDA(cudaFree(spp)); CUDA(cudaFree(spc)); CUDA(cudaFree(spr)); CUDA(cudaFree(spv));
    for(int block_size: {16,64,256,0,-1,-4,-8,-32}) {
        auto block_start=Clock::now(); auto t=block_size>0?pack_blocked(a,block_size):pack_components(a,-block_size); auto block_end=Clock::now();
        int *bo=upload(t.block_order),*bp=upload(t.block_phase),*pgp=upload(t.phase_group),*pw=upload(t.phase_width);
        int *gp=upload(t.group_ptr),*gr=upload(t.group_row),*rr=upload(t.rows),*cc=upload(t.col); double* vv=upload(t.val);
        auto g=capture(stream,[&]{for(size_t w=0;w+1<t.wave_ptr.size();++w) {
            int offset=t.wave_ptr[w], count=t.wave_ptr[w+1]-offset;
            blocked_packed_kernel<<<count,128,0,stream>>>(bo,offset,bp,pgp,pw,gp,gr,rr,cc,vv,dp,dv,db,dx);
            CUDA(cudaGetLastError());
        }});
        CUDA(cudaMemsetAsync(dx,0,a.n*sizeof(double),stream)); CUDA(cudaGraphLaunch(g.exec,stream));
        error=check_result("blocked adaptive"); us=measure(g,stream); destroy(g);
        std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<','<<(block_size>0?"blocked-"+std::to_string(block_size):block_size<0?"components-"+std::to_string(-block_size):"components")<<','<<std::chrono::duration<double,std::micro>(block_end-block_start).count()<<','<<us<<','<<error<<','<<backward<<'\n';
        std::cerr<<"blocked="<<block_size<<",waves="<<t.wave_ptr.size()-1<<",blocks="<<t.block_order.size()<<",phases="<<t.phase_width.size()<<",slots="<<t.val.size()<<'\n';
        if(block_size<=0) {
            auto wg=capture(stream,[&]{component_warp_kernel<<<t.block_order.size(),32,0,stream>>>(bo,bp,pgp,pw,gp,gr,rr,cc,vv,dp,dv,db,dx);CUDA(cudaGetLastError());});
            CUDA(cudaMemsetAsync(dx,0,a.n*sizeof(double),stream));CUDA(cudaGraphLaunch(wg.exec,stream));error=check_result("component warp");us=measure(wg,stream);error=std::max(error,check_result("component warp after replay"));destroy(wg);
            std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<','<<(block_size<0?"components-warp-"+std::to_string(-block_size):"components-warp")<<','<<std::chrono::duration<double,std::micro>(block_end-block_start).count()<<','<<us<<','<<error<<','<<backward<<'\n';
        }
        if(block_size<=0) {
            auto run=[&]{blocked_packed_kernel<<<t.block_order.size(),128,0,stream>>>(bo,0,bp,pgp,pw,gp,gr,rr,cc,vv,dp,dv,db,dx);CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());};
            for(int i=0;i<5;i++) run(); auto host_start=Clock::now(); for(int i=0;i<200;i++) run(); auto host_end=Clock::now();
            error=check_result("component host");
            std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<','<<(block_size<0?"components-host-"+std::to_string(-block_size):"components-host")<<','<<std::chrono::duration<double,std::micro>(block_end-block_start).count()<<','<<std::chrono::duration<double,std::micro>(host_end-host_start).count()/200<<','<<error<<','<<backward<<'\n';
        }
        CUDA(cudaFree(bo)); CUDA(cudaFree(bp)); CUDA(cudaFree(pgp)); CUDA(cudaFree(pw)); CUDA(cudaFree(gp)); CUDA(cudaFree(gr)); CUDA(cudaFree(rr)); CUDA(cudaFree(cc)); CUDA(cudaFree(vv));
    }
    auto dep_start=Clock::now(); auto deps=make_dependents(a); auto dep_end=Clock::now();
    int *ddep=upload(deps.ptr),*drows=upload(deps.rows),*dinitial=upload(deps.initial),*dtemplate=upload(deps.indegree);
    int *ddegree,*head,*tail; QueueSlot* queue;
    CUDA(cudaMalloc(&ddegree,a.n*sizeof(int))); CUDA(cudaMalloc(&head,sizeof(int))); CUDA(cudaMalloc(&tail,sizeof(int)));
    CUDA(cudaMalloc(&queue,a.n*sizeof(QueueSlot)));
    int device,sm,occupancy; CUDA(cudaGetDevice(&device)); CUDA(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,device));
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,queue_solve,128,0));
    // All producer and consumer blocks must fit concurrently; spinning blocks
    // must never prevent a producer block from becoming resident.
    int blocks=std::max(1,sm);
    auto qg=capture(stream,[&]{
        CUDA(cudaMemsetAsync(queue,0,a.n*sizeof(QueueSlot),stream));
        CUDA(cudaMemcpyAsync(ddegree,dtemplate,a.n*sizeof(int),cudaMemcpyDeviceToDevice,stream));
        init_queue<<<std::max(1,int((deps.initial.size()+127)/128)),128,0,stream>>>(queue,dinitial,deps.initial.size(),head,tail);
        // Cooperative launch guarantees simultaneous grid residency.
        void* args[]={&dp,&dc,&dv,&ddep,&drows,&ddegree,&queue,&head,&tail,&a.n,&db,&dx};
        CUDA(cudaLaunchCooperativeKernel((const void*)queue_solve,dim3(blocks),dim3(128),args,0,stream));
    });
    CUDA(cudaMemsetAsync(dx,0,a.n*sizeof(double),stream)); CUDA(cudaGraphLaunch(qg.exec,stream));
    error=check_result("queue"); us=measure(qg,stream); error=std::max(error,check_result("queue after replay")); check_alternate(qg); destroy(qg);
    std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<",queue,"<<std::chrono::duration<double,std::micro>(dep_end-dep_start).count()<<','<<us<<','<<error<<','<<backward<<'\n';
    std::cerr<<"queue_blocks="<<blocks<<",initial="<<deps.initial.size()<<'\n';
    double *ddepval=upload(deps.values),*partial; int* push_next;
    CUDA(cudaMalloc(&partial,a.n*sizeof(double)));CUDA(cudaMalloc(&push_next,sizeof(int)));
    for(int grid: {4,16,32,sm}) for(int sleep_ns: {0,32}) {
        auto pg=capture(stream,[&]{
            CUDA(cudaMemsetAsync(partial,0,a.n*sizeof(double),stream));CUDA(cudaMemsetAsync(push_next,0,sizeof(int),stream));
            CUDA(cudaMemcpyAsync(ddegree,dtemplate,a.n*sizeof(int),cudaMemcpyDeviceToDevice,stream));
            void* args[]={&dp,&dv,&ddep,&drows,&ddepval,&dr,&push_next,&ddegree,&partial,&a.n,&db,&dx,&sleep_ns};
            CUDA(cudaLaunchCooperativeKernel((const void*)push_solve,dim3(grid),dim3(128),args,0,stream));
        });
        CUDA(cudaMemsetAsync(dx,0,a.n*sizeof(double),stream));CUDA(cudaGraphLaunch(pg.exec,stream));error=check_result("push");
        us=measure(pg,stream);error=std::max(error,check_result("push after replay"));check_alternate(pg);destroy(pg);
        std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<",push-"<<grid<<"-sleep-"<<sleep_ns<<','<<std::chrono::duration<double,std::micro>(dep_end-dep_start).count()<<','<<us<<','<<error<<','<<backward<<'\n';
    }
    CUDA(cudaFree(ddepval));CUDA(cudaFree(partial));CUDA(cudaFree(push_next));
    CUDA(cudaFree(ddep)); CUDA(cudaFree(drows)); CUDA(cudaFree(dinitial)); CUDA(cudaFree(dtemplate)); CUDA(cudaFree(ddegree)); CUDA(cudaFree(head)); CUDA(cudaFree(tail)); CUDA(cudaFree(queue));
    int *ord_next,*ord_ready;CUDA(cudaMalloc(&ord_next,sizeof(int)));CUDA(cudaMalloc(&ord_ready,a.n*sizeof(int)));
    for(int width: {0,1,4,8,16,32}) {
        auto og=capture(stream,[&]{
            CUDA(cudaMemsetAsync(ord_ready,0,a.n*sizeof(int),stream));CUDA(cudaMemsetAsync(ord_next,0,sizeof(int),stream));
            void* args[]={&dp,&dc,&dv,&dr,&ord_next,&ord_ready,&a.n,&db,&dx,&width};
            CUDA(cudaLaunchCooperativeKernel((const void*)ordered_solve,dim3(sm),dim3(128),args,0,stream));
        });
        CUDA(cudaMemsetAsync(dx,0,a.n*sizeof(double),stream));CUDA(cudaGraphLaunch(og.exec,stream));
        error=check_result("ordered");us=measure(og,stream);error=std::max(error,check_result("ordered after replay"));check_alternate(og);destroy(og);
        std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<','<<(width?"ordered-"+std::to_string(width):"ordered-adaptive")<<','<<std::chrono::duration<double,std::micro>(end-begin).count()<<','<<us<<','<<error<<','<<backward<<'\n';
    }
    for(int grid_blocks: {1,4,16,32}) {
        int width=4;
        auto og=capture(stream,[&]{
            CUDA(cudaMemsetAsync(ord_ready,0,a.n*sizeof(int),stream));CUDA(cudaMemsetAsync(ord_next,0,sizeof(int),stream));
            void* args[]={&dp,&dc,&dv,&dr,&ord_next,&ord_ready,&a.n,&db,&dx,&width};
            CUDA(cudaLaunchCooperativeKernel((const void*)ordered_solve,dim3(grid_blocks),dim3(128),args,0,stream));
        });
        CUDA(cudaMemsetAsync(dx,0,a.n*sizeof(double),stream));CUDA(cudaGraphLaunch(og.exec,stream));
        error=check_result("ordered grid");us=measure(og,stream);error=std::max(error,check_result("ordered grid after replay"));check_alternate(og);destroy(og);
        std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<",ordered-grid-"<<grid_blocks<<','<<std::chrono::duration<double,std::micro>(end-begin).count()<<','<<us<<','<<error<<','<<backward<<'\n';
    }
    CUDA(cudaFree(ord_next));CUDA(cudaFree(ord_ready));
    int* doff=upload(levels.offsets);
    for(int width: {1,4,32}) {
        auto g=capture(stream,[&]{
            switch(width) {
            case 1: solve_single_cta<1><<<1,256,0,stream>>>(dp,dc,dv,dr,doff,levels.width.size(),db,dx); break;
            case 4: solve_single_cta<4><<<1,256,0,stream>>>(dp,dc,dv,dr,doff,levels.width.size(),db,dx); break;
            case 32: solve_single_cta<32><<<1,256,0,stream>>>(dp,dc,dv,dr,doff,levels.width.size(),db,dx); break;
            } CUDA(cudaGetLastError());
        });
        CUDA(cudaMemsetAsync(dx,0,a.n*sizeof(double),stream)); CUDA(cudaGraphLaunch(g.exec,stream)); error=check_result("single CTA"); us=measure(g,stream); destroy(g);
        std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<",single-cta-"<<width<<','<<std::chrono::duration<double,std::micro>(end-begin).count()<<','<<us<<','<<error<<','<<backward<<'\n';
    }
    if(size_t(a.n)*sizeof(double)<=size_t(shared_limit)) {
        size_t bytes=size_t(a.n)*sizeof(double);
        CUDA(cudaFuncSetAttribute(solve_shared_csr<1>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
        CUDA(cudaFuncSetAttribute(solve_shared_csr<4>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
        CUDA(cudaFuncSetAttribute(solve_shared_csr<8>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
        CUDA(cudaFuncSetAttribute(solve_shared_csr<32>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
        for(int width: {1,4,8,32}) for(int threads: {128,256,512}) {
            auto sg=capture(stream,[&]{switch(width) {
#define SHARED_CASE(W) case W: solve_shared_csr<W><<<1,threads,bytes,stream>>>(dp,dc,dv,dr,doff,levels.width.size(),a.n,db,dx);break
                SHARED_CASE(1);SHARED_CASE(4);SHARED_CASE(8);SHARED_CASE(32);
#undef SHARED_CASE
            } CUDA(cudaGetLastError());});
            CUDA(cudaMemsetAsync(dx,0,a.n*sizeof(double),stream));CUDA(cudaGraphLaunch(sg.exec,stream));error=check_result("shared CSR");
            us=measure(sg,stream);error=std::max(error,check_result("shared CSR replay"));check_alternate(sg);destroy(sg);
            std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<",shared-csr-"<<width<<'-'<<threads<<','<<std::chrono::duration<double,std::micro>(end-begin).count()<<','<<us<<','<<error<<','<<backward<<'\n';
        }
    }
    size_t cached_bytes=sizeof(double)*(size_t(2)*a.n+a.val.size())+sizeof(int)*(a.ptr.size()+a.col.size()+levels.rows.size()+levels.offsets.size());
    if(cached_bytes<=size_t(shared_limit)) {
        CUDA(cudaFuncSetAttribute(solve_cached_csr<1>,cudaFuncAttributeMaxDynamicSharedMemorySize,cached_bytes));
        CUDA(cudaFuncSetAttribute(solve_cached_csr<4>,cudaFuncAttributeMaxDynamicSharedMemorySize,cached_bytes));
        CUDA(cudaFuncSetAttribute(solve_cached_csr<8>,cudaFuncAttributeMaxDynamicSharedMemorySize,cached_bytes));
        CUDA(cudaFuncSetAttribute(solve_cached_csr<32>,cudaFuncAttributeMaxDynamicSharedMemorySize,cached_bytes));
        for(int width: {1,4,8,32}) for(int threads: {128,256,512}) {
            auto cg=capture(stream,[&]{switch(width) {
#define CACHED_CASE(W) case W: solve_cached_csr<W><<<1,threads,cached_bytes,stream>>>(dp,dc,dv,dr,doff,levels.width.size(),a.n,a.val.size(),db,dx);break
                CACHED_CASE(1);CACHED_CASE(4);CACHED_CASE(8);CACHED_CASE(32);
#undef CACHED_CASE
            } CUDA(cudaGetLastError());});
            CUDA(cudaMemsetAsync(dx,0,a.n*sizeof(double),stream));CUDA(cudaGraphLaunch(cg.exec,stream));error=check_result("cached CSR");us=measure(cg,stream);
            error=std::max(error,check_result("cached CSR replay"));check_alternate(cg);destroy(cg);
            std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<",cached-csr-"<<width<<'-'<<threads<<','<<std::chrono::duration<double,std::micro>(end-begin).count()<<','<<us<<','<<error<<','<<backward<<'\n';
        }
    }
    CUDA(cudaFree(doff));
    for(int mode: {1,2,4,8,16,32,0}) {
        auto g=capture(stream,[&]{for(size_t s=0;s<levels.width.size();s++) {int width=mode?mode:levels.width[s],start=levels.offsets[s],count=levels.offsets[s+1]-start; launch(width,count,start,dp,dc,dv,dr,db,dx,stream);}});
        CUDA(cudaMemsetAsync(dx,0,a.n*sizeof(double),stream)); CUDA(cudaGraphLaunch(g.exec,stream)); error=check_result("level"); us=measure(g,stream); destroy(g);
        std::cout<<argv[1]<<','<<a.n<<','<<a.val.size()<<','<<levels.width.size()<<','<<(mode?std::to_string(mode):"adaptive")<<','<<std::chrono::duration<double,std::micro>(end-begin).count()<<','<<us<<','<<error<<','<<backward<<'\n';
    }
    } // !cached_only
    SP(cusparseSpSV_destroyDescr(desc)); SP(cusparseDestroyDnVec(b)); SP(cusparseDestroyDnVec(x)); SP(cusparseDestroySpMat(mat)); SP(cusparseDestroy(h));
    CUDA(cudaFree(buf)); CUDA(cudaFree(dp)); CUDA(cudaFree(dc)); CUDA(cudaFree(dr)); CUDA(cudaFree(dv)); CUDA(cudaFree(db)); CUDA(cudaFree(dx)); CUDA(cudaStreamDestroy(stream));
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;} }
