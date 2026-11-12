#include "sptrsvPlan.h"
#include "csrReader.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
using Clock=std::chrono::steady_clock;
int main(int argc,char**argv){try{
    if(argc!=2 && !(argc==3 && std::string(argv[2])=="--reverse"))throw std::invalid_argument("usage: sptrsv_plan_bench factor.mtx [--reverse]");
    auto source=read_mtx(argv[1]);if(source.nrows!=source.ncols || source.nrows<=0)throw std::runtime_error("square matrix required");
    atspmv::SptrsvHostCsr a;a.rows=source.nrows;a.row_ptr.assign(source.row_ptr,source.row_ptr+a.rows+1);
    a.col_idx.assign(source.col_idx,source.col_idx+source.nnz);a.values.assign(source.val,source.val+source.nnz);
    delete[] source.row_ptr;delete[] source.col_idx;delete[] source.val;
    std::vector<double> truth(a.rows),rhs(a.rows,0),output(a.rows);
    for(int i=0;i<a.rows;i++)truth[i]=.5+((i*37LL)%101)/101.;
    for(int i=0;i<a.rows;i++)for(int p=a.row_ptr[i];p<a.row_ptr[i+1];p++)rhs[i]+=a.values[p]*truth[a.col_idx[p]];
    double *db,*dx;check(cudaMalloc(&db,a.rows*sizeof(double)));check(cudaMalloc(&dx,a.rows*sizeof(double)));
    check(cudaMemcpy(db,rhs.data(),a.rows*sizeof(double),cudaMemcpyHostToDevice));cudaStream_t stream;check(cudaStreamCreate(&stream));
    {auto warm=atspmv::makeSptrsvPlan(a,db,dx,{},stream);atspmv::executeSptrsv(warm,stream);check(cudaStreamSynchronize(stream));}
    std::cout<<std::setprecision(10)<<"matrix,mode,backend,rows,nnz,levels,setup_us,solve_us,max_error\n";
    for(int iteration=0;iteration<2;iteration++) {
        int mode=argc==3?1-iteration:iteration;
        atspmv::SptrsvPlanOptions options;options.enable_cached_tile_model=mode;
        auto start=Clock::now();auto plan=atspmv::makeSptrsvPlan(a,db,dx,options,stream);auto end=Clock::now();
        auto setup=std::chrono::duration<double,std::micro>(end-start).count();
        cudaGraph_t graph;cudaGraphExec_t exec;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        atspmv::executeSptrsv(plan,stream);check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        check(cudaMemsetAsync(dx,0,a.rows*sizeof(double),stream));check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(output.data(),dx,a.rows*sizeof(double),cudaMemcpyDeviceToHost));double error=0.;
        for(int i=0;i<a.rows;i++){if(!std::isfinite(output[i]))throw std::runtime_error("nonfinite result");error=std::max(error,std::abs(output[i]-truth[i]));}
        // Long double normalized backward error supports ill-conditioned factors.
        long double rn=0,an=0,xn=0,bn=0;
        for(int i=0;i<a.rows;i++){long double sum=0,row=0;for(int p=a.row_ptr[i];p<a.row_ptr[i+1];p++){sum+=(long double)a.values[p]*output[a.col_idx[p]];row+=std::abs((long double)a.values[p]);}rn=std::max(rn,std::abs(sum-rhs[i]));an=std::max(an,row);xn=std::max(xn,std::abs((long double)output[i]));bn=std::max(bn,std::abs((long double)rhs[i]));}
        if(rn>1e-12L*(an*xn+bn))throw std::runtime_error("backward error");
        for(int i=0;i<5;i++)check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));cudaEvent_t b,e;check(cudaEventCreate(&b));check(cudaEventCreate(&e));std::vector<float> times;
        for(int k=0;k<9;k++){check(cudaEventRecord(b,stream));for(int i=0;i<10;i++)check(cudaGraphLaunch(exec,stream));check(cudaEventRecord(e,stream));check(cudaEventSynchronize(e));float ms;check(cudaEventElapsedTime(&ms,b,e));times.push_back(ms*100.);}
        std::sort(times.begin(),times.end());auto info=plan.info();
        std::cout<<argv[1]<<','<<(mode?"model":"default")<<','<<(info.backend==atspmv::SptrsvBackend::CachedTile?"cached":"cusparse")<<','<<a.rows<<','<<a.values.size()<<','<<info.levels<<','<<setup<<','<<times[times.size()/2]<<','<<error<<'\n';
        check(cudaEventDestroy(b));check(cudaEventDestroy(e));check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));
    }
    check(cudaStreamDestroy(stream));check(cudaFree(db));check(cudaFree(dx));return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
