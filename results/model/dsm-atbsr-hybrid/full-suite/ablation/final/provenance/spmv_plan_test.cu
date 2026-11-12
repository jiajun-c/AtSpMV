#include "spmvPlan.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
using namespace atspmv;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool condition,const char* msg){if(!condition)throw std::runtime_error(msg);}
template<class F> void rejects(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}require(caught,"expected rejection");}
template<class T> T* upload(const std::vector<T>& v){T* p;check(cudaMalloc(&p,std::max(size_t(1),v.size())*sizeof(T)));if(!v.empty())check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
struct Matrix {
    std::vector<int> ptr,col;std::vector<double> val;
    CSRMatrix<double,int32_t> device;
    Matrix(int rows,int length,bool mixed,int cols=8192) {
        device.nrows=rows;device.ncols=cols;ptr.push_back(0);
        for(int r=0;r<rows;++r) {
            const int count=mixed?(r%5==0?length-r%17:r%71):length;
            for(int j=0;j<std::max(0,count);++j){col.push_back((j*17LL+r*31)%cols);val.push_back(((j*13LL+r*7)%101-50)/53.);}
            ptr.push_back(col.size());
        }
        device.nnz=col.size();device.row_ptr=upload(ptr);device.col_idx=upload(col);device.val=upload(val);
    }
    void release(){if(device.row_ptr)cudaFree(device.row_ptr);if(device.col_idx)cudaFree(device.col_idx);if(device.val)cudaFree(device.val);device.row_ptr=nullptr;device.col_idx=nullptr;device.val=nullptr;}
    ~Matrix(){release();}
};
int checks=0;
void verifyPlan(SpmvPlan& p,const Matrix& a,cudaStream_t stream) {
    const int m=a.device.nrows,n=a.device.ncols;
    std::vector<double> x(n),y(m);double* dx=upload(x);double* dy=upload(y);
    cudaGraph_t graph;cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for(int j=0;j<3;++j)executeSpmv(p,dx,dy,stream);
    check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    for(int changed=0;changed<2;++changed) {
        for(int j=0;j<n;++j)x[j]=((j*(changed?31LL:23LL)+7)%113-56)/67.;
        check(cudaMemcpyAsync(dx,x.data(),n*sizeof(double),cudaMemcpyHostToDevice,stream));
        for(int mode=0;mode<2;++mode) {
            check(cudaMemsetAsync(dy,0xff,size_t(m)*sizeof(double),stream));
            if(mode)check(cudaGraphLaunch(executable,stream));else executeSpmv(p,dx,dy,stream);
            check(cudaStreamSynchronize(stream));check(cudaMemcpy(y.data(),dy,m*sizeof(double),cudaMemcpyDeviceToHost));
            for(int r=0;r<m;++r) {
                long double sum=0,mag=0;
                for(int j=a.ptr[r];j<a.ptr[r+1];++j){long double v=(long double)a.val[j]*x[a.col[j]];sum+=v;mag+=std::abs(v);}
                long double bound=8*std::numeric_limits<double>::epsilon()*(a.ptr[r+1]-a.ptr[r]+32.L)*mag+1e-12L;
                require(std::isfinite(y[r])&&std::abs(y[r]-sum)<=bound,"incorrect planned SpMV");
            }
            ++checks;
        }
    }
    if(m&&n){rejects([&]{executeSpmv(p,dx,dx,stream);});rejects([&]{executeSpmv(p,dx,dx+1,stream);});}
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaFree(dx));check(cudaFree(dy));
}
int main(int argc,char** argv){try{
    const bool quick=argc==2&&std::string(argv[1])=="--quick";
    if(argc>2||(argc==2&&!quick))throw std::invalid_argument("Usage: spmv_plan_test [--quick]");
    static_assert(!std::is_copy_constructible_v<SpmvPlan>);
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    SpmvPlan empty;rejects([&]{empty.info();});rejects([&]{executeSpmv(empty,nullptr,nullptr,stream);});
    Matrix a(quick?9:137,quick?1031:4103,true);
    SpmvPlanOptions defaults;auto base=makeSpmvPlan(a.device,defaults,stream);
    require(base.info().backend==SpmvBackend::AtBsr&&!base.info().measured,"default must preserve model selection");
    const bool dsm=base.info().dsm_available;verifyPlan(base,a,stream);
    for(auto backend:{SpmvBackend::AtBsr,SpmvBackend::CsrTwoStage,SpmvBackend::CsrDsm}) {
        if(backend==SpmvBackend::CsrDsm&&!dsm) {
            auto options=defaults;options.backend=backend;rejects([&]{makeSpmvPlan(a.device,options,stream);});continue;
        }
        for(int group:{1,4,8})for(int threshold:{0,1024,100000}) {
            auto options=defaults;options.backend=backend;options.long_rows={8,256,group,threshold};
            options.tile_index=2;options.balanced=group==1?0:1;
            auto p=makeSpmvPlan(a.device,options,stream);verifyPlan(p,a,stream);
        }
    }
    for(auto backend:{SpmvBackend::CsrTwoStage,SpmvBackend::CsrDsm}) {
        if(backend==SpmvBackend::CsrDsm&&!dsm)continue;
        Matrix small(9,1031,true);
        for(int threads:{512,1024}) {
            auto o=defaults;o.backend=backend;o.long_rows={8,threads,8,1024};
            auto p=makeSpmvPlan(small.device,o,stream);verifyPlan(p,small,stream);
        }
    }
    if(dsm) {
        for(int tile=0;tile<6;++tile)for(int threshold:{0,1024,100000}) {
            auto o=defaults;o.backend=SpmvBackend::HybridDsmAtBsr;o.tile_index=tile;
            o.long_rows={8,256,4,threshold};auto p=makeSpmvPlan(a.device,o,stream);verifyPlan(p,a,stream);
        }
        Matrix empty_short(9,1031,true);
        auto o=defaults;o.backend=SpmvBackend::HybridDsmAtBsr;o.long_rows={4,128,1,1};
        auto p=makeSpmvPlan(empty_short.device,o,stream);verifyPlan(p,empty_short,stream);
        o.tune=true;o.tile_index=2;o.measurement_batches=2;o.calls_per_batch=2;
        o.tuning_long_rows={{4,128,1,1024},{8,256,4,1024}};
        auto tuned=makeSpmvPlan(a.device,o,stream);require(tuned.info().measured,"hybrid tune missing");verifyPlan(tuned,a,stream);
        { // Auto skips hybrid candidates whose partition contains no DSM rows; explicit hybrid retains them.
            SpmvPlanOptions gate;gate.tune=true;gate.tile_index=2;gate.measurement_batches=2;gate.calls_per_batch=2;
            auto automatic=makeSpmvPlan(a.device,gate,stream);
            int max_length=0;for(size_t r=1;r<a.ptr.size();++r)max_length=std::max(max_length,a.ptr[r]-a.ptr[r-1]);
            for(const auto& c:automatic.info().timings)if(c.backend==SpmvBackend::HybridDsmAtBsr)
                require(c.long_rows.threshold<=max_length,"auto retained empty DSM partition");
            gate.tuning_long_rows={{4,128,1,100000}};
            auto custom=makeSpmvPlan(a.device,gate,stream);bool preserved=false;
            for(const auto& c:custom.info().timings)if(c.backend==SpmvBackend::HybridDsmAtBsr)preserved=true;
            require(preserved,"user-specified candidate removed");
            gate.backend=SpmvBackend::HybridDsmAtBsr;gate.tile_index=2;
            auto explicit_hybrid=makeSpmvPlan(a.device,gate,stream);
            require(explicit_hybrid.info().timings.size()==1,"explicit hybrid candidate removed");
            verifyPlan(explicit_hybrid,a,stream);
        }
        {
            Matrix short_rows(9,3,false);SpmvPlanOptions o;o.tune=true;o.tile_index=2;
            o.measurement_batches=2;o.calls_per_batch=2;
            auto p=makeSpmvPlan(short_rows.device,o,stream);int empty_candidates=0;
            for(const auto& c:p.info().timings)if(c.backend==SpmvBackend::HybridDsmAtBsr&&c.long_rows.threshold>3) {
                ++empty_candidates;require(c.long_rows.threshold==1024&&c.long_rows.parts==8&&c.long_rows.threads==256&&c.long_rows.rows_per_group==1,"unexpected no-long candidate");
            }
            require(empty_candidates==1,"no-long representative missing");verifyPlan(p,short_rows,stream);
        }
        { // Different thresholds selecting identical rows share packing safely.
            SpmvPlanOptions shared;shared.tune=true;shared.backend=SpmvBackend::HybridDsmAtBsr;
            shared.measurement_batches=2;shared.calls_per_batch=2;shared.tile_index=2;
            shared.tuning_long_rows={{4,128,1,128},{4,128,1,512}};
            auto packed=makeSpmvPlan(a.device,shared,stream);
            require(packed.info().timings.size()==2,"hybrid threshold candidates missing");verifyPlan(packed,a,stream);
        }
        { // Lower thresholds are proposed only when they create a distinct medium-row partition.
            Matrix medium(9,500,true);SpmvPlanOptions o;o.tune=true;o.tile_index=2;
            o.measurement_batches=2;o.calls_per_batch=2;
            auto p=makeSpmvPlan(medium.device,o,stream);bool found128=false,found256=false;
            for(const auto& c:p.info().timings)if(c.backend==SpmvBackend::HybridDsmAtBsr) {
                found128|=c.long_rows.threshold==128;found256|=c.long_rows.threshold==256;
            }
            require(found128&&found256,"medium-row hybrid candidates missing");verifyPlan(p,medium,stream);
        }
        Matrix wide(17,1031,true,65536);o.tune=false;o.tile_index=3;o.long_rows={2,256,8,1024};
        auto wide_plan=makeSpmvPlan(wide.device,o,stream);wide.release();verifyPlan(wide_plan,wide,stream);
        o.balanced=1;rejects([&]{makeSpmvPlan(a.device,o,stream);});
    } else {
        auto o=defaults;o.backend=SpmvBackend::HybridDsmAtBsr;
        rejects([&]{makeSpmvPlan(a.device,o,stream);});
    }
    for(auto backend:{SpmvBackend::Auto,SpmvBackend::AtBsr,SpmvBackend::CsrTwoStage,SpmvBackend::CsrDsm}) {
        if(backend==SpmvBackend::CsrDsm&&!dsm)continue;
        SpmvPlanOptions options;options.backend=backend;options.tune=true;
        options.warmup_batches=1;options.measurement_batches=3;options.calls_per_batch=2;
        options.tuning_long_rows={{4,128,1,1024},{8,256,4,4096},{32,128,1,1024}};
        auto p=makeSpmvPlan(a.device,options,stream);
        require(p.info().measured&&!p.info().timings.empty(),"missing timing report");
        auto best=std::min_element(p.info().timings.begin(),p.info().timings.end(),[](auto& l,auto& r){return l.median_us<r.median_us;});
        require(best->backend==p.info().backend&&best->tile_index==p.info().tile_index&&best->balanced==p.info().balanced,"wrong tuning winner");
        if(best->backend!=SpmvBackend::AtBsr)require(best->long_rows.parts==p.info().long_rows.parts&&best->long_rows.threads==p.info().long_rows.threads&&best->long_rows.rows_per_group==p.info().long_rows.rows_per_group&&best->long_rows.threshold==p.info().long_rows.threshold,"wrong long-row configuration");
        std::cout<<"tuned "<<spmvBackendName(backend)<<" -> "<<spmvBackendName(p.info().backend)<<", candidates="<<p.info().timings.size()<<", setup_us="<<p.info().setup_us<<'\n';
        auto moved=std::move(p);rejects([&]{p.info();});verifyPlan(moved,a,stream);
    }
    // Explicitly test ownership for every backend after input buffers die.
    for(auto backend:{SpmvBackend::AtBsr,SpmvBackend::CsrTwoStage,SpmvBackend::CsrDsm}) {
        if(backend==SpmvBackend::CsrDsm&&!dsm)continue;
        Matrix owned(17,1031,true);auto options=defaults;options.backend=backend;
        auto p=makeSpmvPlan(owned.device,options,stream);owned.release();verifyPlan(p,owned,stream);
        // Move assignment releases an existing populated plan safely.
        base=std::move(p);verifyPlan(base,owned,stream);
    }
    { // Default-stream construction must also support measured selection.
        SpmvPlanOptions o;o.tune=true;o.backend=SpmvBackend::CsrTwoStage;
        o.measurement_batches=2;o.tuning_long_rows={{4,128,1,1024}};
        auto p=makeSpmvPlan(a.device,o);verifyPlan(p,a,stream);
    }
    for(int rows:{0,7}) {
        Matrix zero(rows,0,false,0);auto options=defaults;options.tune=true;
        auto p=makeSpmvPlan(zero.device,options,stream);require(!p.info().measured,"zero nnz should bypass tuning");verifyPlan(p,zero,stream);
    }
    auto bad=defaults;bad.long_rows.rows_per_group=3;rejects([&]{makeSpmvPlan(a.device,bad,stream);});
    bad=defaults;bad.measurement_batches=0;rejects([&]{makeSpmvPlan(a.device,bad,stream);});
    Matrix malformed(3,5,false);int invalid=-1;check(cudaMemcpy(malformed.device.col_idx,&invalid,sizeof(int),cudaMemcpyHostToDevice));
    rejects([&]{makeSpmvPlan(malformed.device,defaults,stream);});
    // Explicit stream capture must reject construction before allocations.
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    rejects([&]{makeSpmvPlan(a.device,defaults,stream);});
    cudaGraph_t graph;check(cudaStreamEndCapture(stream,&graph));check(cudaGraphDestroy(graph));
    int devices=0;check(cudaGetDeviceCount(&devices));
    if(devices>1) {
        int owner;check(cudaGetDevice(&owner));check(cudaSetDevice((owner+1)%devices));
        rejects([&]{executeSpmv(base,nullptr,nullptr);});
        base=SpmvPlan{};int after;check(cudaGetDevice(&after));
        require(after==(owner+1)%devices,"destructor failed to restore current device");
        check(cudaSetDevice(owner));
    }
    check(cudaStreamDestroy(stream));
    std::cout<<"PASS: "<<checks<<" direct/graph/reference checks; dsm_available="<<dsm<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
