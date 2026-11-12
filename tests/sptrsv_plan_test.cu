#include "sptrsvPlan.h"
#include "detail/sptrsvPacking.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool b,const char* msg){if(!b)throw std::runtime_error(msg);}
void run(bool chain,int size=128,bool enable=true) {
    using namespace atspmv;
    SptrsvHostCsr a;a.rows=size;a.row_ptr.push_back(0);
    for(int i=0;i<a.rows;i++) {
        if(chain && i){a.col_idx.push_back(i-1);a.values.push_back(-.2);}
        a.col_idx.push_back(i);a.values.push_back(1.+i%3);a.row_ptr.push_back(a.values.size());
    }
    double *db,*dx;check(cudaMalloc(&db,a.rows*sizeof(double)));check(cudaMalloc(&dx,a.rows*sizeof(double)));
    cudaStream_t stream;check(cudaStreamCreate(&stream));
    {
        SptrsvPlanOptions options;options.enable_cached_tile_model=enable;
        auto p=makeSptrsvPlan(a,db,dx,options,stream);
        if(!chain || !enable || size>4000)require(p.info().backend==SptrsvBackend::CuSparse,"expected fallback");
        if(p.info().levels>0) {
            sptrsv_detail::Matrix m{a.rows,a.row_ptr,a.col_idx,a.values};
            auto levels=sptrsv_detail::make_levels(m);auto packed=sptrsv_detail::pack_levels(m,levels);
            auto groups=packed.ptr.size()-1;
            std::size_t expected=sizeof(double)*(std::size_t(3)*a.rows+packed.val.size())+
                sizeof(int)*(packed.ptr.size()+packed.col.size()+groups+packed.rows.size()+packed.group_offsets.size()+levels.width.size());
            if(p.info().backend==SptrsvBackend::CachedCsr)expected=sizeof(double)*(std::size_t(2)*a.rows+a.values.size())+
                sizeof(int)*(a.row_ptr.size()+a.col_idx.size()+levels.rows.size()+levels.offsets.size());
            require(expected==p.info().cached_bytes,"incorrect cached size estimate");
        }
        cudaDeviceProp prop{};int dev;check(cudaGetDevice(&dev));check(cudaGetDeviceProperties(&prop,dev));
        if(chain && enable && size<=4000 && prop.major==9 && prop.multiProcessorCount==114 && std::string(prop.name).find("H100 PCIe")!=std::string::npos)
            {require(p.info().backend==SptrsvBackend::CachedCsr,"chain should select cached CSR");require(p.info().cached_threads==256,"chain should use 256 threads");}
        auto verify=[&](int phase,bool graph){
            std::vector<double> truth(a.rows),rhs(a.rows,0),out(a.rows);
            for(int i=0;i<a.rows;i++)truth[i]=.25+((i*37+phase*13)%97)/97.;
            for(int i=0;i<a.rows;i++)for(int j=a.row_ptr[i];j<a.row_ptr[i+1];j++)rhs[i]+=a.values[j]*truth[a.col_idx[j]];
            check(cudaMemcpyAsync(db,rhs.data(),a.rows*sizeof(double),cudaMemcpyHostToDevice,stream));check(cudaMemsetAsync(dx,0,a.rows*sizeof(double),stream));
            if(graph){cudaGraph_t g;cudaGraphExec_t exec;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));executeSptrsv(p,stream);check(cudaStreamEndCapture(stream,&g));check(cudaGraphInstantiate(&exec,g,nullptr,nullptr,0));for(int i=0;i<3;i++)check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(g));}
            else {executeSptrsv(p,stream);check(cudaStreamSynchronize(stream));}
            check(cudaMemcpy(out.data(),dx,a.rows*sizeof(double),cudaMemcpyDeviceToHost));
            for(int i=0;i<a.rows;i++)require(std::isfinite(out[i]) && std::abs(out[i]-truth[i])<1e-12,"incorrect solution");
        };
        verify(1,false);verify(2,true);verify(3,false);
    }
    check(cudaStreamDestroy(stream));check(cudaFree(db));check(cudaFree(dx));
}
void multiple_plans() {
    using namespace atspmv;
    auto matrix=[](int n){SptrsvHostCsr a;a.rows=n;a.row_ptr.push_back(0);for(int i=0;i<n;i++){if(i){a.col_idx.push_back(i-1);a.values.push_back(-.25);}a.col_idx.push_back(i);a.values.push_back(1.);a.row_ptr.push_back(a.values.size());}return a;};
    auto big=matrix(256),small=matrix(64);std::vector<double> rhs(256,.75),out(256);rhs[0]=1.;
    double *big_b,*big_x,*small_b,*small_x;check(cudaMalloc(&big_b,256*sizeof(double)));check(cudaMalloc(&big_x,256*sizeof(double)));
    check(cudaMalloc(&small_b,64*sizeof(double)));check(cudaMalloc(&small_x,64*sizeof(double)));
    check(cudaMemcpy(big_b,rhs.data(),256*sizeof(double),cudaMemcpyHostToDevice));check(cudaMemcpy(small_b,rhs.data(),64*sizeof(double),cudaMemcpyHostToDevice));
    {
        SptrsvPlanOptions opt;opt.enable_cached_tile_model=true;
        auto large=makeSptrsvPlan(big,big_b,big_x,opt);auto tiny=makeSptrsvPlan(small,small_b,small_x,opt);
        executeSptrsv(large);executeSptrsv(tiny);check(cudaDeviceSynchronize());
        check(cudaMemcpy(out.data(),big_x,256*sizeof(double),cudaMemcpyDeviceToHost));
        for(double x:out)require(std::abs(x-1.)<1e-12,"live plans interfere");
    }
    check(cudaFree(big_b));check(cudaFree(big_x));check(cudaFree(small_b));check(cudaFree(small_x));
}
// Exercise all tile widths, incomplete row groups and rows longer than one
// tile, with signed non-unit diagonals and a different RHS on each replay.
void varied_levels() {
    using namespace atspmv;
    SptrsvHostCsr a; a.row_ptr.push_back(0);
    const int levels=11, rows_per_level=35;
    for(int level=0;level<levels;++level) {
        const int first=level*rows_per_level;
        const int lengths[]={0,1,3,8,24,80,1,3,8,24,80};
        for(int r=0;r<rows_per_level;++r) {
            int count=lengths[level];
            for(int j=0;j<count;++j) {
                // The last dependency is in the immediately previous level,
                // so every constructed level has the intended dependency depth.
                int col=j==count-1 ? first-1 : j;
                a.col_idx.push_back(col);
                a.values.push_back(((r+j)%2 ? -.02 : .02));
            }
            a.col_idx.push_back(first+r);
            a.values.push_back((r%2 ? -1. : 1.)*(2.+.03*count+.01*r));
            a.row_ptr.push_back(a.values.size());
        }
    }
    a.rows=levels*rows_per_level;
    sptrsv_detail::Matrix m{a.rows,a.row_ptr,a.col_idx,a.values};
    auto packing=sptrsv_detail::make_levels(m);
    for(int width : {1,4,8,16,32})
        require(std::find(packing.width.begin(),packing.width.end(),width)!=packing.width.end(),"tile width missing from test");
    double *b,*x; check(cudaMalloc(&b,a.rows*sizeof(double))); check(cudaMalloc(&x,a.rows*sizeof(double)));
    cudaStream_t stream; check(cudaStreamCreate(&stream));
    {
        SptrsvPlanOptions options; options.enable_cached_tile_model=true;
        auto plan=makeSptrsvPlan(a,b,x,options,stream);
        cudaDeviceProp prop{}; int device; check(cudaGetDevice(&device)); check(cudaGetDeviceProperties(&prop,device));
        if(prop.major==9 && prop.multiProcessorCount==114 && std::string(prop.name).find("H100 PCIe")!=std::string::npos)
            {require(plan.info().backend==SptrsvBackend::CachedTile,"varied levels should select cached tile");require(plan.info().cached_threads==1024,"wide levels should use 1024 threads");}
        cudaGraph_t graph; cudaGraphExec_t exec;
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal)); executeSptrsv(plan,stream);
        check(cudaStreamEndCapture(stream,&graph)); check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        for(int phase=0;phase<5;++phase) {
            std::vector<double> truth(a.rows),rhs(a.rows,0.),out(a.rows);
            for(int i=0;i<a.rows;++i)truth[i]=((i*37+phase*13)%101-50)/37.;
            for(int i=0;i<a.rows;++i)for(int j=a.row_ptr[i];j<a.row_ptr[i+1];++j)rhs[i]+=a.values[j]*truth[a.col_idx[j]];
            check(cudaMemcpyAsync(b,rhs.data(),a.rows*sizeof(double),cudaMemcpyHostToDevice,stream));
            check(cudaMemsetAsync(x,0xff,a.rows*sizeof(double),stream));
            if(phase%2)check(cudaGraphLaunch(exec,stream));else executeSptrsv(plan,stream);
            check(cudaMemcpyAsync(out.data(),x,a.rows*sizeof(double),cudaMemcpyDeviceToHost,stream));
            check(cudaStreamSynchronize(stream));
            for(int i=0;i<a.rows;++i)require(std::isfinite(out[i]) && std::abs(out[i]-truth[i])<1e-11,"varied levels solution incorrect");
        }
        check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph));
    }
    check(cudaStreamDestroy(stream)); check(cudaFree(b)); check(cudaFree(x));
}
void dense_chain() {
    using namespace atspmv;
    SptrsvHostCsr a;a.rows=160;a.row_ptr.push_back(0);
    for(int i=0;i<a.rows;++i) {
        for(int j=std::max(0,i-1-(i%23));j<i;++j){a.col_idx.push_back(j);a.values.push_back((j%2 ? -.02 : .02));}
        a.col_idx.push_back(i);a.values.push_back((i%2 ? -1. : 1.)*(2.+.01*i));a.row_ptr.push_back(a.values.size());
    }
    double *b,*x;check(cudaMalloc(&b,a.rows*sizeof(double)));check(cudaMalloc(&x,a.rows*sizeof(double)));
    cudaStream_t stream;check(cudaStreamCreate(&stream));
    {
        SptrsvPlanOptions options;options.enable_cached_tile_model=true;auto plan=makeSptrsvPlan(a,b,x,options,stream);
        cudaDeviceProp prop{};int device;check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&prop,device));
        if(prop.major==9 && prop.multiProcessorCount==114 && std::string(prop.name).find("H100 PCIe")!=std::string::npos)
            require(plan.info().backend==SptrsvBackend::CachedCsr,"dense chain should select cached CSR");
        cudaGraph_t graph;cudaGraphExec_t exec;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));executeSptrsv(plan,stream);
        check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        for(int phase=0;phase<4;++phase) {
            std::vector<double> truth(a.rows),rhs(a.rows,0.),out(a.rows);
            for(int i=0;i<a.rows;++i)truth[i]=((i*37+phase*13)%101-50)/37.;
            for(int i=0;i<a.rows;++i)for(int j=a.row_ptr[i];j<a.row_ptr[i+1];++j)rhs[i]+=a.values[j]*truth[a.col_idx[j]];
            check(cudaMemcpyAsync(b,rhs.data(),a.rows*sizeof(double),cudaMemcpyHostToDevice,stream));check(cudaMemsetAsync(x,0xff,a.rows*sizeof(double),stream));
            if(phase%2)check(cudaGraphLaunch(exec,stream));else executeSptrsv(plan,stream);
            check(cudaMemcpyAsync(out.data(),x,a.rows*sizeof(double),cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
            for(int i=0;i<a.rows;++i)require(std::isfinite(out[i]) && std::abs(out[i]-truth[i])<1e-11,"dense chain solution incorrect");
        }
        check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));
    }
    check(cudaStreamDestroy(stream));check(cudaFree(b));check(cudaFree(x));
}
void invalid() {
    using namespace atspmv;
    double *v;check(cudaMalloc(&v,64*sizeof(double)));
    SptrsvHostCsr a{2,{0,1,2},{0,1},{1.,1.}};
    auto must_reject=[&](const SptrsvHostCsr& m,const double* b,double* x){bool rejected=false;try{auto p=makeSptrsvPlan(m,b,x);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"invalid input accepted");};
    must_reject(a,v,v+1);auto zero=a;zero.values[1]=0.;must_reject(zero,v,v+8);
    auto unsorted=a;unsorted.row_ptr={0,1,3};unsorted.col_idx={0,1,0};unsorted.values={1.,1.,1.};must_reject(unsorted,v,v+8);
    check(cudaFree(v));
}
int main(){try{run(true);run(false);run(true,128,false);run(false,4097,true);multiple_plans();varied_levels();dense_chain();invalid();std::cout<<"SpTRSV plan direct/Graph correctness passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
