#include "spmvPlan.h"
#include "atspmvprod.h"
#include <algorithm>
#include <cmath>
#include <climits>
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
    { // Every packed candidate owns its row permutation after shared sorting
      // buffers are freed; exercise all tiles and both index widths.
        for(int columns:{8192,65536}) {
            Matrix source(31,47,true,columns);
            std::vector<int> zeros(source.device.nrows);int* lengths=upload(zeros);int* indices=upload(zeros);
            prepareAtBSRProductRowOrder(source.device,lengths,indices);
            AtBSRProductStatsCache stats;
            std::vector<AtBsr<double,int32_t>> packed(12);
            for(int tile=0;tile<6;++tile)for(int balance=0;balance<2;++balance) {
                auto& candidate=packed[2*tile+balance];
                CSRToAtBSRProductFromRows(source.device,candidate,tile,balance,lengths,indices,&stats);
                AtBsr<double,int32_t> uncached;
                CSRToAtBSRProduct(source.device,uncached,tile,balance);
                require(candidate.blockCount==uncached.blockCount&&candidate.warpCount==uncached.warpCount&&
                        candidate.modelCalibrated==uncached.modelCalibrated&&
                        std::abs(candidate.predictedTimeUs-uncached.predictedTimeUs)<=1e-12*(1+std::abs(uncached.predictedTimeUs)),
                        "cached tile statistics changed model predictions");
                for(void* ptr:{static_cast<void*>(uncached.blcVal),static_cast<void*>(uncached.blcIdx),static_cast<void*>(uncached.blcIdxInt16),static_cast<void*>(uncached.blcPtr),static_cast<void*>(uncached.recorder),static_cast<void*>(uncached.targetRowByWarp),static_cast<void*>(uncached.warpBlcPtr)})if(ptr)check(cudaFree(ptr));
                if(tile==0&&balance==0) {stats.tiles_per_task=0;stats.stats[0].tiles=ULLONG_MAX;}
                if(tile==0&&balance==1)require(stats.tiles_per_task!=0&&stats.stats[0].tiles!=ULLONG_MAX,"tile stats cache was not invalidated");
            }
            require(stats.initialized,"shared tile statistics were not captured");
            check(cudaFree(lengths));check(cudaFree(indices));
            std::vector<double> x(columns),y(source.device.nrows);double* dx=upload(x);double* dy=upload(y);
            for(int changed=0;changed<2;++changed) {
                for(int col=0;col<columns;++col)x[col]=((col*(changed?31LL:23LL)+7)%113-56)/67.;
                check(cudaMemcpy(dx,x.data(),x.size()*sizeof(double),cudaMemcpyHostToDevice));
                for(auto& candidate:packed) {
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
        if(dsm&&prop.major==9&&prop.minor==0&&prop.multiProcessorCount==114&&prop.l2CacheSize==52428800&&std::string(prop.name)=="NVIDIA H100 PCIe") {
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
