#include "spmvPlan.h"
#include "atspmvprod.h"
#include <thrust/system/detail/bad_alloc.h>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdlib>
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
    { // Mixed layouts own their storage and cover both index widths, empty
      // rows, direct execution, repeated graph launches and all task caps.
        for(int columns:{8192,65536}) {
            Matrix source(37,1301,true,columns);
            for(int cap:{-1,0,4,8,16}) {
                SpmvPlanOptions o;o.backend=SpmvBackend::MixedTile;o.mixed_tile_cap=cap;
                auto plan=makeSpmvPlan(source.device,o,stream);
                require(plan.info().backend==SpmvBackend::MixedTile&&plan.info().mixed_tile_cap==cap,"explicit mixed plan configuration");
                verifyPlan(plan,source,stream);
            }
            SpmvPlanOptions tuned;tuned.backend=SpmvBackend::MixedTile;tuned.tune=true;tuned.measurement_batches=2;tuned.calls_per_batch=2;
            auto plan=makeSpmvPlan(source.device,tuned,stream);
            require(plan.info().measured&&plan.info().timings.size()==5,"mixed tuning must compare five modes");
            verifyPlan(plan,source,stream);
        }
        SpmvPlanOptions auto_mixed;auto_mixed.tune=true;auto_mixed.fast_tune=true;auto_mixed.enable_mixed_tile=true;
        auto_mixed.measurement_batches=2;auto_mixed.calls_per_batch=2;
        auto plan=makeSpmvPlan(a.device,auto_mixed,stream);
        int count=0;for(const auto& candidate:plan.info().timings)if(candidate.backend==SpmvBackend::MixedTile)++count;
        require(count==5,"auto tuning must include five mixed candidates");verifyPlan(plan,a,stream);
        for(bool all_empty:{false,true}) {
            Matrix source(17,0,!all_empty);
            for(int cap:{-1,0,4,8,16}) {
                SpmvPlanOptions o;o.backend=SpmvBackend::MixedTile;o.mixed_tile_cap=cap;
                auto plan=makeSpmvPlan(source.device,o,stream);
                verifyPlan(plan,source,stream);
            }
        }
        { Matrix source(23,400,true);SpmvPlanOptions o;o.backend=SpmvBackend::MixedTile;o.mixed_tile_cap=8;
          auto owned=makeSpmvPlan(source.device,o,stream);source.release();verifyPlan(owned,source,stream); }
        SpmvPlanOptions invalid;invalid.backend=SpmvBackend::MixedTile;invalid.mixed_tile_cap=3;
        rejects([&]{makeSpmvPlan(a.device,invalid,stream);});
    }

    { // Failed tile selection must release row-order allocations.
        Matrix source(200000,1,false);
        const char* old=std::getenv("ATSPMV_MODEL_CACHE");
        const bool had_old=old!=nullptr;const std::string saved=old?old:"";
        require(setenv("ATSPMV_MODEL_CACHE","invalid-test-value",1)==0,"cannot set invalid model environment");
        auto fail=[&]{AtBsr<double,int32_t> packed;rejects([&]{CSRToAtBSRProduct(source.device,packed,-1,-1);});};
        fail();check(cudaDeviceSynchronize());
        size_t before=0,after=0,total=0;check(cudaMemGetInfo(&before,&total));
        for(int i=0;i<16;++i)fail();
        check(cudaDeviceSynchronize());check(cudaMemGetInfo(&after,&total));
        if(had_old)require(setenv("ATSPMV_MODEL_CACHE",saved.c_str(),1)==0,"cannot restore model environment");
        else require(unsetenv("ATSPMV_MODEL_CACHE")==0,"cannot restore model environment");
        require(after+4*1024*1024>=before,"failed tile selection leaked GPU memory");
    }
    if(std::getenv("ATSPMV_ENABLE_FAIL_MALLOC_TEST")) {
        { // Failed mixed packing must fall back to the remaining measured
          // candidates without leaking its partially allocated buffers.
            Matrix source(1000,1,false);
            SpmvPlanOptions o;o.model_select=true;o.model_refine=true;o.enable_mixed_tile=true;
            o.measurement_batches=2;o.calls_per_batch=2;
            require(setenv("ATSPMV_INJECT_CUDA_MALLOC","512:0",1)==0,"cannot enable mixed allocation fault");
            auto recovered=makeSpmvPlan(source.device,o,stream);
            require(unsetenv("ATSPMV_INJECT_CUDA_MALLOC")==0,"cannot disable mixed allocation fault");
            bool skipped=false;
            for(const auto& item:recovered.info().skipped_candidates)
                skipped|=item.find("mixed_tile: insufficient GPU memory")!=std::string::npos;
            require(skipped&&recovered.info().measured,"mixed allocation failure did not fall back");
            for(const auto& item:recovered.info().timings)
                require(item.backend!=SpmvBackend::MixedTile,"failed mixed candidate was retained");
            verifyPlan(recovered,source,stream);
            recovered=SpmvPlan();check(cudaDeviceSynchronize());
            size_t before=0,after=0,total=0;check(cudaMemGetInfo(&before,&total));
            for(int i=0;i<4;++i) {
                require(setenv("ATSPMV_INJECT_CUDA_MALLOC","512:0",1)==0,"cannot enable repeated mixed allocation fault");
                auto plan=makeSpmvPlan(source.device,o,stream);
                require(unsetenv("ATSPMV_INJECT_CUDA_MALLOC")==0,"cannot disable repeated mixed allocation fault");
                require(!plan.info().skipped_candidates.empty(),"repeated mixed fault was not triggered");
            }
            check(cudaDeviceSynchronize());check(cudaMemGetInfo(&after,&total));
            require(after+4*1024*1024>=before,"failed mixed packing leaked GPU memory");
        }
        // Run with tests/cuda_malloc_fault.c preloaded. For tile m=32 the
        // second 25004-byte allocation is rowPtrbyWarp, after packed arrays.
        Matrix source(200000,1,false);
        { // The third allocation happens inside Thrust's row-order sort.
            AtBsr<double,int32_t> packed;bool thrust_oom=false;
            require(setenv("ATSPMV_INJECT_CUDA_MALLOC","0:2",1)==0,"cannot enable Thrust allocation fault");
            try {CSRToAtBSRProduct(source.device,packed,5,1);}
            catch(const thrust::system::detail::bad_alloc& e) {thrust_oom=std::string(e.what()).find(cudaGetErrorString(cudaErrorMemoryAllocation))!=std::string::npos;}
            catch(const std::exception&) {}
            require(unsetenv("ATSPMV_INJECT_CUDA_MALLOC")==0,"cannot disable Thrust allocation fault");
            require(thrust_oom,"allocation fault did not reach Thrust's allocator");
        }
        check(cudaDeviceSynchronize());size_t before=0,after=0,total=0;
        check(cudaMemGetInfo(&before,&total));
        for(int i=0;i<8;++i) {
            AtBsr<double,int32_t> packed;
            require(setenv("ATSPMV_INJECT_CUDA_MALLOC","25004:1",1)==0,"cannot enable allocation fault");
            bool failed=false;
            try {CSRToAtBSRProduct(source.device,packed,5,1);}
            catch(const std::exception& e) {failed=std::string(e.what()).find("out of memory")!=std::string::npos;}
            require(unsetenv("ATSPMV_INJECT_CUDA_MALLOC")==0,"cannot disable allocation fault");
            require(failed,"injected CUDA allocation failure was not reported");
        }
        check(cudaDeviceSynchronize());check(cudaMemGetInfo(&after,&total));
        require(after+4*1024*1024>=before,"failed packed allocation leaked GPU memory");
        {
            SpmvPlanOptions o;o.backend=SpmvBackend::AtBsr;o.tune=true;
            o.retry_shared_storage_on_oom=true;o.measurement_batches=2;o.calls_per_batch=2;
            require(setenv("ATSPMV_INJECT_CUDA_MALLOC","0:5",1)==0&&setenv("ATSPMV_INJECT_CUDA_UNKNOWN","1",1)==0,"cannot enable unrelated Thrust allocation fault");
            rejects([&]{makeSpmvPlan(source.device,o,stream);});
            require(unsetenv("ATSPMV_INJECT_CUDA_MALLOC")==0&&unsetenv("ATSPMV_INJECT_CUDA_UNKNOWN")==0,"cannot disable unrelated Thrust allocation fault");
            require(setenv("ATSPMV_INJECT_CUDA_MALLOC","0:5",1)==0,"cannot enable tuned Thrust allocation fault");
            SpmvPlan thrust_recovered;
            try {thrust_recovered=makeSpmvPlan(source.device,o,stream);}
            catch(const thrust::system::detail::bad_alloc& e) {throw std::runtime_error(std::string("tuned Thrust bad_alloc: ")+e.what());}
            catch(const std::exception& e) {throw std::runtime_error(std::string("tuned Thrust retry: ")+e.what());}
            require(unsetenv("ATSPMV_INJECT_CUDA_MALLOC")==0,"cannot disable tuned Thrust allocation fault");
            require(thrust_recovered.info().oom_retry_shared_storage_used&&thrust_recovered.info().timings.size()==12,
                    "Thrust allocation failure did not retry all AtBSR candidates");
            verifyPlan(thrust_recovered,source,stream);
            require(setenv("ATSPMV_INJECT_CUDA_MALLOC","25004:1",1)==0,"cannot enable allocation fault");
            auto recovered=makeSpmvPlan(source.device,o,stream);
            require(unsetenv("ATSPMV_INJECT_CUDA_MALLOC")==0,"cannot disable allocation fault");
            require(recovered.info().oom_retry_shared_storage_used&&recovered.info().timings.size()==12,
                    "allocation failure did not retry all AtBSR candidates");
            verifyPlan(recovered,source,stream);
        }
        { // The optional out-of-scope model fallback preserves retry info.
            Matrix wide(4096,1,false,1785346);
            SpmvPlanOptions o;o.model_select=true;o.model_fallback_tune=true;
            o.retry_shared_storage_on_oom=true;o.measurement_batches=2;o.calls_per_batch=2;
            require(setenv("ATSPMV_INJECT_CUDA_MALLOC","516:1",1)==0,"cannot enable fallback allocation fault");
            auto recovered=makeSpmvPlan(wide.device,o,stream);
            require(unsetenv("ATSPMV_INJECT_CUDA_MALLOC")==0,"cannot disable fallback allocation fault");
            require(recovered.info().model_fallback_tune_used&&recovered.info().oom_retry_shared_storage_used,
                    "model fallback did not preserve allocation retry info");
            verifyPlan(recovered,wide,stream);
            SpmvPlanOptions sparse;sparse.tune=true;sparse.measurement_batches=2;sparse.calls_per_batch=2;
            require(setenv("ATSPMV_INJECT_CUSPARSE_CREATE","1",1)==0,"cannot enable cuSPARSE allocation fault");
            rejects([&]{makeSpmvPlan(wide.device,sparse,stream);});
            sparse.retry_shared_storage_on_oom=true;
            require(setenv("ATSPMV_INJECT_CUSPARSE_CREATE","other",1)==0,"cannot enable unrelated cuSPARSE fault");
            rejects([&]{makeSpmvPlan(wide.device,sparse,stream);});
            require(setenv("ATSPMV_INJECT_CUSPARSE_CREATE","1",1)==0,"cannot re-enable cuSPARSE allocation fault");
            auto sparse_recovered=makeSpmvPlan(wide.device,sparse,stream);
            require(sparse_recovered.info().oom_retry_shared_storage_used,"cuSPARSE allocation failure did not retry");
            verifyPlan(sparse_recovered,wide,stream);
        }
    }
    { // Every packed candidate owns its row permutation after shared sorting
      // buffers are freed; exercise all tiles and both index widths.
        for(int columns:{8192,65536}) {
            Matrix source(31,47,true,columns);
            std::vector<int> zeros(source.device.nrows);int* lengths=upload(zeros);int* indices=upload(zeros);
            prepareAtBSRProductRowOrder(source.device,lengths,indices);
            AtBSRProductStatsCache stats;
            const char* prior_cap=std::getenv("ATSPMV_TUNE_CAP");
            const std::string saved_cap=prior_cap?prior_cap:"";const bool cap_set=prior_cap!=nullptr;
            const int changed_cap=atspmv::kernelConfigFromEnvironment().tiles_per_task==4?16:4;
            std::vector<AtBsr<double,int32_t>> packed(12);
            for(int tile=0;tile<6;++tile)for(int balance=0;balance<2;++balance) {
                if(tile==0&&balance==1)require(setenv("ATSPMV_TUNE_CAP",std::to_string(changed_cap).c_str(),1)==0,"cannot change tile task capacity");
                auto& candidate=packed[2*tile+balance];
                CSRToAtBSRProductFromRows(source.device,candidate,tile,balance,lengths,indices,&stats);
                AtBsr<double,int32_t> uncached;
                CSRToAtBSRProduct(source.device,uncached,tile,balance);
                require(candidate.blockCount==uncached.blockCount&&candidate.warpCount==uncached.warpCount&&
                        candidate.modelCalibrated==uncached.modelCalibrated&&
                        std::abs(candidate.predictedTimeUs-uncached.predictedTimeUs)<=1e-12*(1+std::abs(uncached.predictedTimeUs)),
                        "cached tile statistics changed model predictions");
                for(void* ptr:{static_cast<void*>(uncached.blcVal),static_cast<void*>(uncached.blcIdx),static_cast<void*>(uncached.blcIdxInt16),static_cast<void*>(uncached.blcPtr),static_cast<void*>(uncached.recorder),static_cast<void*>(uncached.targetRowByWarp),static_cast<void*>(uncached.warpBlcPtr)})if(ptr)check(cudaFree(ptr));
                if(tile==0&&balance==1) {
                    require(stats.tiles_per_task==changed_cap,"tile stats cache was not invalidated");
                    if(cap_set)require(setenv("ATSPMV_TUNE_CAP",saved_cap.c_str(),1)==0,"cannot restore tile task capacity");
                    else require(unsetenv("ATSPMV_TUNE_CAP")==0,"cannot restore tile task capacity");
                }
            }
            require(stats.initialized,"shared tile statistics were not captured");
            check(cudaFree(lengths));check(cudaFree(indices));
            // Normal-mode execution over the balanced packing must remain
            // correct for every shape and both index widths.
            std::vector<AtBsr<double,int32_t>> shared(6);
            for(int tile=0;tile<6;++tile) {
                shared[tile]=packed[2*tile+1];
                shared[tile].useBalance=false;
                shared[tile].warpCount=shared[tile].brow;
            }
            std::vector<double> x(columns),y(source.device.nrows);double* dx=upload(x);double* dy=upload(y);
            for(int changed=0;changed<2;++changed) {
                for(int col=0;col<columns;++col)x[col]=((col*(changed?31LL:23LL)+7)%113-56)/67.;
                check(cudaMemcpy(dx,x.data(),x.size()*sizeof(double),cudaMemcpyHostToDevice));
                for(int i=0;i<18;++i) {
                    auto& candidate=i<12?packed[i]:shared[i-12];
                    check(cudaMemsetAsync(dy,0xff,y.size()*sizeof(double),stream));
                    AtSpmvProduct(candidate,dx,dy,1,0,stream);check(cudaStreamSynchronize(stream));
                    check(cudaMemcpy(y.data(),dy,y.size()*sizeof(double),cudaMemcpyDeviceToHost));
                    for(int row=0;row<source.device.nrows;++row) {
                        long double sum=0,mag=0;
                        for(int j=source.ptr[row];j<source.ptr[row+1];++j){long double value=(long double)source.val[j]*x[source.col[j]];sum+=value;mag+=std::abs(value);}
                        require(std::isfinite(y[row])&&std::abs(y[row]-sum)<=8*std::numeric_limits<double>::epsilon()*(source.ptr[row+1]-source.ptr[row]+32.L)*mag+1e-12L,"incorrect shared-order packed candidate");
                    }
                    ++checks;
                }
            }
            check(cudaFree(dx));check(cudaFree(dy));
            for(auto& candidate:packed)for(void* ptr:{static_cast<void*>(candidate.blcVal),static_cast<void*>(candidate.blcIdx),static_cast<void*>(candidate.blcIdxInt16),static_cast<void*>(candidate.blcPtr),static_cast<void*>(candidate.recorder),static_cast<void*>(candidate.targetRowByWarp),static_cast<void*>(candidate.warpBlcPtr)})if(ptr)check(cudaFree(ptr));
        }
    }
    { // Shared packed candidates retain their storage after tuning cleanup.
        SpmvPlanOptions o;o.backend=SpmvBackend::AtBsr;o.tune=true;
        o.share_atbsr_tuning_storage=true;o.measurement_batches=2;o.calls_per_batch=2;
        auto p=makeSpmvPlan(a.device,o,stream);
        require(p.info().measured&&p.info().timings.size()==12,"shared AtBSR tuning lost candidates");
        verifyPlan(p,a,stream);
    }
    {
        SpmvPlanOptions fast;fast.tune=true;fast.fast_tune=true;fast.measurement_batches=2;fast.calls_per_batch=2;
        auto p=makeSpmvPlan(a.device,fast,stream);
        require(p.info().measured&&p.info().fast_tuning_used,"fast tuning was not used");
        require(p.info().timings.size()<=21,"fast shortlist unexpectedly large");verifyPlan(p,a,stream);
        fast.tile_index=2;auto forced=makeSpmvPlan(a.device,fast,stream);
        require(!forced.info().fast_tuning_used,"forced tile must retain full tuning semantics");verifyPlan(forced,a,stream);
        fast.tile_index=-1;fast.tuning_long_rows={{4,128,1,1024}};
        auto custom=makeSpmvPlan(a.device,fast,stream);
        require(!custom.info().fast_tuning_used,"custom candidates must retain full tuning semantics");verifyPlan(custom,a,stream);
    }
    {
        SpmvPlanOptions model;model.model_select=true;
        auto p=makeSpmvPlan(a.device,model,stream);
        cudaDeviceProp prop{};int device;check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&prop,device));
        if(dsm&&prop.major==9&&prop.minor==0&&prop.multiProcessorCount==114&&prop.l2CacheSize==52428800&&std::string(prop.name)=="NVIDIA H100 PCIe"&&kernelConfigFromEnvironment().baseline()) {
            require(p.info().model_used&&!p.info().model_predictions.empty(),"H100 model was not used");
            require(p.info().model_reference_us>0,"missing reference estimate");
            for(const auto& c:p.info().model_predictions)
                require(std::abs(c.estimated_us-c.time_ratio_to_default*p.info().model_reference_us)<1e-10,"wrong absolute estimate");
            const SpmvModelPrediction* best=nullptr;double estimate=1.;
            for(const auto& c:p.info().model_predictions)if(c.backend!=SpmvBackend::HybridDsmAtBsr&&c.time_ratio_to_default<.95&&c.time_ratio_to_default<estimate){best=&c;estimate=c.time_ratio_to_default;}
            const SpmvModelPrediction* hybrid=nullptr;double gate=estimate*.9;
            for(const auto& c:p.info().model_predictions)if(c.backend==SpmvBackend::HybridDsmAtBsr&&c.time_ratio_to_default<gate){hybrid=&c;gate=c.time_ratio_to_default;}
            if(hybrid)best=hybrid;
            require(p.info().backend==(best?best->backend:SpmvBackend::AtBsr),"model choice disagrees with prediction gates");
            if(best)require(p.info().tile_index==best->tile_index&&p.info().balanced==best->balanced,"model configuration mismatch");
        } else require(!p.info().model_used&&p.info().backend==SpmvBackend::AtBsr,"unsupported hardware must fall back");
        verifyPlan(p,a,stream);
        Matrix short_model(9,3,false);auto short_plan=makeSpmvPlan(short_model.device,model,stream);
        for(const auto& c:short_plan.info().model_predictions)require(c.backend!=SpmvBackend::HybridDsmAtBsr,"model included a no-long hybrid");
        require(short_plan.info().backend!=SpmvBackend::HybridDsmAtBsr,"model selected a no-long hybrid");
        verifyPlan(short_plan,short_model,stream);
        Matrix all_long(9,1031,false);auto long_plan=makeSpmvPlan(all_long.device,model,stream);
        for(const auto& c:long_plan.info().model_predictions)require(c.backend!=SpmvBackend::HybridDsmAtBsr,"model included a long-row-dominated hybrid");
        verifyPlan(long_plan,all_long,stream);
        model.tile_index=2;auto forced=makeSpmvPlan(a.device,model,stream);
        require(!forced.info().model_used&&forced.info().backend==SpmvBackend::AtBsr,"forced tile must disable model choice");
    }
    { // Explicit opt-in vector predictions share the existing model scope.
        SpmvPlanOptions model;model.model_select=true;model.vector_model_select=true;
        {
            auto graph = model; graph.vector_model_graph=true;
            auto plan = makeSpmvPlan(a.device,graph,stream);
            require(plan.info().vector_model_graph_used==plan.info().model_used,"graph model scope mismatch");
            if(plan.info().vector_model_graph_used) {
                int count=0;
                for(const auto& prediction:plan.info().model_predictions)
                    if(prediction.backend==SpmvBackend::CsrVector) {
                        require(std::isfinite(prediction.time_ratio_to_default)&&prediction.time_ratio_to_default>0,"invalid graph prediction");
                        ++count;
                    }
                require(count==6,"missing graph width predictions");
            }
            verifyPlan(plan,a,stream);
            graph.vector_model_v2=true;
            bool rejected=false;
            try { auto invalid=makeSpmvPlan(a.device,graph,stream); }
            catch(const std::invalid_argument&) { rejected=true; }
            require(rejected,"conflicting vector models must be rejected");
        }
        {
            auto v2 = model;v2.vector_model_v2=true;
            auto plan = makeSpmvPlan(a.device,v2,stream);
            require(plan.info().vector_model_v2_used==plan.info().model_used,"v2 scope mismatch");
            if(plan.info().vector_model_v2_used) {
                int count=0;
                for(const auto& prediction:plan.info().model_predictions)
                    if(prediction.backend==SpmvBackend::CsrVector) {
                        require(std::isfinite(prediction.time_ratio_to_default)&&prediction.time_ratio_to_default>0,"invalid v2 prediction");
                        ++count;
                    }
                require(count==6,"missing v2 width predictions");
            }
            verifyPlan(plan,a,stream);
            auto absolute=v2;absolute.absolute_reference_model=true;
            auto absolute_plan=makeSpmvPlan(a.device,absolute,stream);
            require(absolute_plan.info().absolute_reference_model_used==absolute_plan.info().model_used,"absolute reference scope mismatch");
            require(absolute_plan.info().backend==plan.info().backend&&absolute_plan.info().tile_index==plan.info().tile_index&&
                    absolute_plan.info().balanced==plan.info().balanced&&absolute_plan.info().vector_width==plan.info().vector_width,
                    "absolute reference changed selected configuration");
            if(absolute_plan.info().absolute_reference_model_used) {
                require(std::isfinite(absolute_plan.info().model_reference_us)&&absolute_plan.info().model_reference_us>0,"invalid absolute reference");
                for(const auto& prediction:absolute_plan.info().model_predictions)
                    require(std::abs(prediction.estimated_us-prediction.time_ratio_to_default*absolute_plan.info().model_reference_us)<1e-10,"invalid absolute estimate");
            }
            verifyPlan(absolute_plan,a,stream);
            auto medium=v2;medium.medium_row_model=true;
            Matrix medium_rows(9,200,false);
            auto medium_plan=makeSpmvPlan(medium_rows.device,medium,stream);
            require(medium_plan.info().medium_row_model_used==medium_plan.info().model_used,"medium model scope mismatch");
            if(medium_plan.info().medium_row_model_used) {
                int count=0;
                for(const auto& prediction:medium_plan.info().model_predictions)
                    if(prediction.backend==SpmvBackend::HybridDsmAtBsr&&prediction.long_rows.threshold==128) {
                        require(std::isfinite(prediction.time_ratio_to_default)&&prediction.time_ratio_to_default>0,"invalid medium prediction");
                        ++count;
                    }
                require(count==12,"missing medium-row predictions");
            }
            verifyPlan(medium_plan,medium_rows,stream);
        }
        { SpmvPlanOptions invalid;invalid.medium_row_model=true;rejects([&]{makeSpmvPlan(a.device,invalid,stream);}); }
        { SpmvPlanOptions invalid;invalid.absolute_reference_model=true;rejects([&]{makeSpmvPlan(a.device,invalid,stream);}); }
        auto selected=makeSpmvPlan(a.device,model,stream);
        if(selected.info().model_used) {
            require(selected.info().vector_model_used,"vector predictor was not used");
            int vector_count=0;double best=std::numeric_limits<double>::infinity();int width=0;
            for(const auto& pred:selected.info().model_predictions)if(pred.backend==SpmvBackend::CsrVector) {
                ++vector_count;if(pred.time_ratio_to_default<best){best=pred.time_ratio_to_default;width=pred.vector_width;}
            }
            require(vector_count==6,"missing vector model predictions");
            if(selected.info().backend==SpmvBackend::CsrVector)require(selected.info().vector_width==width,"incorrect vector width selected");
        } else require(!selected.info().vector_model_used,"vector predictor ran outside calibrated scope");
        verifyPlan(selected,a,stream);
        if(selected.info().model_used) {
            model.model_fallback_tune=true;auto in_scope=makeSpmvPlan(a.device,model,stream);
            require(in_scope.info().model_used&&!in_scope.info().model_fallback_tune_used&&!in_scope.info().measured,"in-scope model should not tune");
            verifyPlan(in_scope,a,stream);
            require(setenv("ATSPMV_MODEL_BW_EFFICIENCY","0.8",1)==0,"cannot set model environment");
            model.fast_tune=true;model.measurement_batches=2;model.calls_per_batch=2;
            auto custom_env=makeSpmvPlan(a.device,model,stream);
            require(custom_env.info().model_fallback_tune_used&&custom_env.info().measured&&!custom_env.info().model_used,"model override did not trigger fallback tuning");
            verifyPlan(custom_env,a,stream);
            require(unsetenv("ATSPMV_MODEL_BW_EFFICIENCY")==0,"cannot restore model environment");
            model.fast_tune=false;model.model_fallback_tune=false;
        }
        model.vector_model_min_time_reduction=.02;
        auto margin=makeSpmvPlan(a.device,model,stream);
        if(margin.info().model_used) {
            double fallback=1.,hybrid=std::numeric_limits<double>::infinity();
            for(const auto& pred:margin.info().model_predictions)
                if(pred.backend!=SpmvBackend::CsrVector&&pred.backend!=SpmvBackend::HybridDsmAtBsr&&pred.time_ratio_to_default<.95&&pred.time_ratio_to_default<fallback)fallback=pred.time_ratio_to_default;
            for(const auto& pred:margin.info().model_predictions)
                if(pred.backend==SpmvBackend::HybridDsmAtBsr&&pred.time_ratio_to_default<fallback*.9&&pred.time_ratio_to_default<hybrid)hybrid=pred.time_ratio_to_default;
            if(std::isfinite(hybrid))fallback=hybrid;
            double best=fallback*.98;int width=0;
            for(const auto& pred:margin.info().model_predictions)if(pred.backend==SpmvBackend::CsrVector&&pred.time_ratio_to_default<best){best=pred.time_ratio_to_default;width=pred.vector_width;}
            require((margin.info().backend==SpmvBackend::CsrVector)==(width!=0),"vector margin decision differs from predictions");
            if(width)require(margin.info().vector_width==width,"vector margin selected wrong width");
        }
        verifyPlan(margin,a,stream);
        model.vector_model_min_time_reduction=.999;
        auto guarded=makeSpmvPlan(a.device,model,stream);
        require(guarded.info().backend!=SpmvBackend::CsrVector,"conservative vector gate ignored");
        verifyPlan(guarded,a,stream);
        model.vector_model_min_time_reduction=1.;rejects([&]{makeSpmvPlan(a.device,model,stream);});
        model.vector_model_min_time_reduction=0.;
        model.tile_index=2;auto forced=makeSpmvPlan(a.device,model,stream);
        require(!forced.info().vector_model_used&&forced.info().backend==SpmvBackend::AtBsr,"forced tile must skip vector predictor");
        model.tile_index=-1;model.model_select=false;rejects([&]{makeSpmvPlan(a.device,model,stream);});
        model.model_select=true;model.tune=true;rejects([&]{makeSpmvPlan(a.device,model,stream);});
    }
    { // Optional model refinement tests the predicted candidate, default
      // AtBSR and all CSR vector widths; mixed candidates remain opt-in.
        SpmvPlanOptions o;o.model_select=true;o.vector_model_select=true;o.model_refine=true;
        o.enable_mixed_tile=true;o.measurement_batches=2;o.calls_per_batch=2;
        auto refined=makeSpmvPlan(a.device,o,stream);
        require(refined.info().model_refined&&refined.info().measured,"bounded refinement did not measure candidates");
        int vectors=0,mixed=0,atbsr=0;
        for(const auto& candidate:refined.info().timings) {
            vectors+=candidate.backend==SpmvBackend::CsrVector;
            mixed+=candidate.backend==SpmvBackend::MixedTile;
            atbsr+=candidate.backend==SpmvBackend::AtBsr;
        }
        require(vectors==6&&mixed==5&&atbsr>=1,"bounded refinement candidate set incomplete");
        verifyPlan(refined,a,stream);
        o.tile_index=2;rejects([&]{makeSpmvPlan(a.device,o,stream);});
        o.tile_index=-1;o.tune=true;rejects([&]{makeSpmvPlan(a.device,o,stream);});
        o.tune=false;o.model_select=false;rejects([&]{makeSpmvPlan(a.device,o,stream);});
    }
    { // The optional out-of-scope policy tunes instead of silently using the
      // default layout. Forced configurations retain their original meaning.
        Matrix wide_scope(5,3,false,1785346);
        SpmvPlanOptions policy;policy.model_select=true;policy.vector_model_select=true;
        auto plain=makeSpmvPlan(wide_scope.device,policy,stream);
        require(!plain.info().model_used&&!plain.info().model_fallback_tune_used&&!plain.info().measured,"out-of-scope default changed");
        policy.model_fallback_tune=true;policy.fast_tune=true;policy.measurement_batches=2;policy.calls_per_batch=2;
        auto fallback=makeSpmvPlan(wide_scope.device,policy,stream);
        require(fallback.info().model_fallback_tune_used&&fallback.info().measured&&!fallback.info().model_used&&fallback.info().fast_tuning_used,"out-of-scope tuning not used");
        verifyPlan(fallback,wide_scope,stream);
        policy.model_refine=true;
        auto refined_fallback=makeSpmvPlan(wide_scope.device,policy,stream);
        require(refined_fallback.info().model_fallback_tune_used&&!refined_fallback.info().model_refined&&refined_fallback.info().measured,
                "out-of-scope refinement did not use tuning fallback");
        verifyPlan(refined_fallback,wide_scope,stream);
        policy.model_refine=false;
        policy.tile_index=2;auto forced=makeSpmvPlan(wide_scope.device,policy,stream);
        require(!forced.info().model_fallback_tune_used&&!forced.info().measured&&forced.info().backend==SpmvBackend::AtBsr,"forced tile triggered fallback tuning");
        policy.tile_index=-1;policy.model_select=false;policy.vector_model_select=false;
        rejects([&]{makeSpmvPlan(wide_scope.device,policy,stream);});
    }
    { // A captured cuSPARSE graph must retain its own vector pointers after
      // the plan's descriptors are reused for a later direct invocation.
        Matrix matrix(8,5,false);SpmvPlanOptions o;o.backend=SpmvBackend::CuSparse;
        auto p=makeSpmvPlan(matrix.device,o,stream);
        std::vector<double> xa(matrix.device.ncols,1.0),xb(matrix.device.ncols,2.0),zero(matrix.device.nrows,0.0),result(zero.size());
        double *da=upload(xa),*db=upload(xb),*ya=upload(zero),*yb=upload(zero);
        cudaGraph_t graph;cudaGraphExec_t exec;
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));executeSpmv(p,da,ya,stream);
        check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        executeSpmv(p,db,yb,stream);check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(result.data(),ya,result.size()*sizeof(double),cudaMemcpyDeviceToHost));
        for(int row=0;row<matrix.device.nrows;++row) {
            double expected=0;for(int j=matrix.ptr[row];j<matrix.ptr[row+1];++j)expected+=matrix.val[j];
            require(std::abs(result[row]-expected)<1e-10,"cuSPARSE graph changed vector pointers");
        }
        check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));
        check(cudaFree(da));check(cudaFree(db));check(cudaFree(ya));check(cudaFree(yb));
    }
    {
        Matrix rows(73,37,true);Matrix long_rows(3,4103,false);
        for(int width:{1,2,4,8,16,32}) {
            SpmvPlanOptions o;o.backend=SpmvBackend::CsrVector;o.vector_width=width;
            auto p=makeSpmvPlan(rows.device,o,stream);
            require(p.info().vector_width==width,"wrong CSR vector width");verifyPlan(p,rows,stream);
            auto long_plan=makeSpmvPlan(long_rows.device,o,stream);verifyPlan(long_plan,long_rows,stream);
        }
        SpmvPlanOptions o;o.backend=SpmvBackend::CsrVector;o.tune=true;o.measurement_batches=2;o.calls_per_batch=2;
        auto p=makeSpmvPlan(rows.device,o,stream);require(p.info().timings.size()==6,"missing CSR vector widths");verifyPlan(p,rows,stream);
    }
    for(auto backend:{SpmvBackend::AtBsr,SpmvBackend::CsrTwoStage,SpmvBackend::CsrDsm,SpmvBackend::CuSparse}) {
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
        { // Mixed long/short paths, partial final clusters, and all-short/all-long partitions.
            Matrix rows(73,1031,true);
            for(int width:{1,2,4,8,16,32})for(int group:{1,4,8})for(int threshold:{0,1024,100000}) {
                SpmvPlanOptions o;o.backend=SpmvBackend::CsrDsmVector;o.vector_width=width;o.long_rows={4,256,group,threshold};
                auto p=makeSpmvPlan(rows.device,o,stream);require(p.info().vector_width==width,"wrong DSM vector width");verifyPlan(p,rows,stream);
            }
            SpmvPlanOptions o;o.backend=SpmvBackend::CsrDsmVector;o.tune=true;o.measurement_batches=2;o.calls_per_batch=2;
            o.tuning_long_rows={{4,128,1,1024}};auto p=makeSpmvPlan(rows.device,o,stream);
            require(p.info().timings.size()==6,"missing DSM vector candidates");verifyPlan(p,rows,stream);
        }
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
                ++empty_candidates;
            }
            require(empty_candidates==0,"auto retained no-long hybrid");verifyPlan(p,short_rows,stream);
        }
        { // Different thresholds selecting identical rows share packing safely.
            SpmvPlanOptions shared;shared.tune=true;shared.backend=SpmvBackend::HybridDsmAtBsr;
            shared.measurement_batches=2;shared.calls_per_batch=2;shared.tile_index=2;
            shared.tuning_long_rows={{4,128,1,128},{4,128,1,512}};
            auto packed=makeSpmvPlan(a.device,shared,stream);
            require(packed.info().timings.size()==2,"hybrid threshold candidates missing");verifyPlan(packed,a,stream);
        }
        { // Exercise every tile with both nonempty and empty short partitions;
          // cached short CSR and sort buffers are released before execution.
            for(int columns:{8192,65536}) {
                Matrix source(17,1031,true,columns);
                for(int tile=0;tile<6;++tile)for(int threshold:{1,1024}) {
                    SpmvPlanOptions cached;cached.backend=SpmvBackend::HybridDsmAtBsr;cached.tune=true;
                    cached.tile_index=tile;cached.measurement_batches=2;cached.calls_per_batch=2;
                    cached.tuning_long_rows={{4,128,1,threshold},{8,256,4,threshold}};
                    auto candidate=makeSpmvPlan(source.device,cached,stream);
                    require(candidate.info().timings.size()==2,"hybrid cached candidates missing");
                    verifyPlan(candidate,source,stream);
                }
            }
        }
        { // Lower thresholds are proposed only when they create a distinct medium-row partition.
            Matrix medium(9,500,true);SpmvPlanOptions o;o.tune=true;o.tile_index=2;
            o.measurement_batches=2;o.calls_per_batch=2;
            auto p=makeSpmvPlan(medium.device,o,stream);bool found128=false,found128_c8=false,found256=false;
            for(const auto& c:p.info().timings)if(c.backend==SpmvBackend::HybridDsmAtBsr) {
                found128|=c.long_rows.threshold==128;found128_c8|=c.long_rows.threshold==128&&c.long_rows.parts==8&&c.long_rows.rows_per_group==4;
                found256|=c.long_rows.threshold==256;
            }
            require(found128&&found128_c8&&found256,"medium-row hybrid candidates missing");verifyPlan(p,medium,stream);
        }
        Matrix wide(17,1031,true,65536);o.tune=false;o.tile_index=3;o.long_rows={2,256,8,1024};
        auto wide_plan=makeSpmvPlan(wide.device,o,stream);wide.release();verifyPlan(wide_plan,wide,stream);
        o.balanced=1;rejects([&]{makeSpmvPlan(a.device,o,stream);});
    } else {
        auto o=defaults;o.backend=SpmvBackend::HybridDsmAtBsr;
        rejects([&]{makeSpmvPlan(a.device,o,stream);});
        o.backend=SpmvBackend::CsrDsmVector;rejects([&]{makeSpmvPlan(a.device,o,stream);});
    }
    for(auto backend:{SpmvBackend::Auto,SpmvBackend::AtBsr,SpmvBackend::CsrTwoStage,SpmvBackend::CsrDsm,SpmvBackend::CuSparse}) {
        if(backend==SpmvBackend::CsrDsm&&!dsm)continue;
        SpmvPlanOptions options;options.backend=backend;options.tune=true;
        options.warmup_batches=1;options.measurement_batches=3;options.calls_per_batch=2;
        options.tuning_long_rows={{4,128,1,1024},{8,256,4,4096},{32,128,1,1024}};
        auto p=makeSpmvPlan(a.device,options,stream);
        require(p.info().measured&&!p.info().timings.empty(),"missing timing report");
        auto best=std::min_element(p.info().timings.begin(),p.info().timings.end(),[](auto& l,auto& r){return l.median_us<r.median_us;});
        if(backend==SpmvBackend::Auto) {
            auto nonhybrid=p.info().timings.end(),hybrid=p.info().timings.end();
            int max_length=0;for(size_t row=0;row+1<a.ptr.size();++row)max_length=std::max(max_length,a.ptr[row+1]-a.ptr[row]);
            for(auto it=p.info().timings.begin();it!=p.info().timings.end();++it) {
                const bool ish=it->backend==SpmvBackend::HybridDsmAtBsr||it->backend==SpmvBackend::CsrDsmVector;
                if(ish&&max_length<it->long_rows.threshold)continue;
                auto& current=ish?hybrid:nonhybrid;
                if(current==p.info().timings.end()||it->median_us<current->median_us)current=it;
            }
            best=nonhybrid;
            if(hybrid!=p.info().timings.end()&&hybrid->median_us<=nonhybrid->median_us*(1-options.hybrid_min_time_reduction))best=hybrid;
        }
        require(best->backend==p.info().backend&&best->tile_index==p.info().tile_index&&best->balanced==p.info().balanced,"wrong tuning winner");
        if(best->backend==SpmvBackend::CsrVector||best->backend==SpmvBackend::CsrDsmVector)require(best->vector_width==p.info().vector_width,"wrong selected vector width");
        if(best->backend!=SpmvBackend::AtBsr)require(best->long_rows.parts==p.info().long_rows.parts&&best->long_rows.threads==p.info().long_rows.threads&&best->long_rows.rows_per_group==p.info().long_rows.rows_per_group&&best->long_rows.threshold==p.info().long_rows.threshold,"wrong long-row configuration");
        std::cout<<"tuned "<<spmvBackendName(backend)<<" -> "<<spmvBackendName(p.info().backend)<<", candidates="<<p.info().timings.size()<<", setup_us="<<p.info().setup_us<<'\n';
        auto moved=std::move(p);rejects([&]{p.info();});verifyPlan(moved,a,stream);
    }
    // Explicitly test ownership for every backend after input buffers die.
    for(auto backend:{SpmvBackend::AtBsr,SpmvBackend::CsrTwoStage,SpmvBackend::CsrDsm,SpmvBackend::CuSparse}) {
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
        options.backend=SpmvBackend::CsrVector;options.tune=false;
        auto vector=makeSpmvPlan(zero.device,options,stream);verifyPlan(vector,zero,stream);
        options=defaults;options.model_select=true;options.vector_model_select=true;options.model_fallback_tune=true;
        auto policy=makeSpmvPlan(zero.device,options,stream);
        require(!policy.info().model_fallback_tune_used&&!policy.info().model_used&&!policy.info().measured,"empty input should bypass model and fallback tuning");
        verifyPlan(policy,zero,stream);
    }
    auto bad=defaults;bad.long_rows.rows_per_group=3;rejects([&]{makeSpmvPlan(a.device,bad,stream);});
    bad=defaults;bad.measurement_batches=0;rejects([&]{makeSpmvPlan(a.device,bad,stream);});
    bad=defaults;bad.hybrid_min_time_reduction=1;rejects([&]{makeSpmvPlan(a.device,bad,stream);});
    bad=defaults;bad.vector_width=3;rejects([&]{makeSpmvPlan(a.device,bad,stream);});
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
