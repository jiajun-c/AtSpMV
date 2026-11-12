// Experimental mixed AtBSR tiles: optimize allocated slots, then compare
// one heterogeneous kernel against each fixed tile/balance combination.
#include "atspmvprod.h"
#include "csrReader.h"
#include "detail/mixedTile.cuh"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> static T* upload(const std::vector<T>& v) {
    T* p=nullptr; ck(cudaMalloc(&p,std::max(size_t(1),v.size())*sizeof(T)));
    if(!v.empty())ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); return p;
}
using atspmv::detail::Mixed;
using atspmv::detail::make_mixed;
struct Candidate {std::string name;AtBsr<double,int32_t> a;int mixed=0;cudaGraph_t graph{};cudaGraphExec_t exec{};cudaEvent_t begin{},end{};std::vector<double> samples;size_t bytes=0;int slots=0;};
static void free_atbsr(AtBsr<double,int32_t>& a) {
    for(void* p:{static_cast<void*>(a.blcVal),static_cast<void*>(a.blcIdx),static_cast<void*>(a.blcIdxInt16),static_cast<void*>(a.blcPtr),static_cast<void*>(a.recorder),static_cast<void*>(a.targetRowByWarp),static_cast<void*>(a.warpBlcPtr)})if(p)ck(cudaFree(p));
}
int main(int argc,char**argv) {try {
    if(argc!=5)throw std::invalid_argument("usage: mixed_tile_bench matrix.mtx seed trials order(0|1)");
    unsigned seed=std::stoul(argv[2]);int trials=std::stoi(argv[3]),order=std::stoi(argv[4]);
    if(trials<1||trials>1000||(order!=0&&order!=1))throw std::invalid_argument("invalid trials/order");
    auto host=read_mtx(argv[1]);int rows=host.nrows,cols=host.ncols,nnz=host.nnz;
    if(rows<=0||cols<=0||nnz<=0)throw std::invalid_argument("experiment requires nonempty matrix");
    std::vector<int> ptr(host.row_ptr,host.row_ptr+rows+1),col(host.col_idx,host.col_idx+nnz);
    std::vector<double> val(host.val,host.val+nnz);delete[] host.row_ptr;delete[] host.col_idx;delete[] host.val;
    CSRMatrix<double,int32_t> a{};a.nrows=rows;a.ncols=cols;a.nnz=nnz;a.row_ptr=upload(ptr);a.col_idx=upload(col);a.val=upload(val);
    std::vector<double> x(cols),y(rows);double *dx=upload(x),*dy=upload(y);
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    std::vector<Candidate> candidates(30);candidates[0].name="mixed";candidates[0].mixed=1;
    candidates[1].name="mixed_specialized";candidates[1].mixed=2;
    candidates[2].name="mixed_bal4";candidates[2].mixed=3;
    candidates[3].name="mixed_bal8";candidates[3].mixed=4;
    candidates[4].name="mixed_bal16";candidates[4].mixed=5;candidates[5].name="default";
    for(int t=0;t<6;++t)for(int b=0;b<2;++b)candidates[6+2*t+b].name="m"+std::to_string(1<<t)+"_b"+std::to_string(b);
    for(int t=0;t<6;++t)for(int j=0;j<2;++j)candidates[18+2*t+j].name="m"+std::to_string(1<<t)+"_b1_cap"+std::to_string(j?16:4);
    Mixed mixed;
    // Both mixed execution variants share exactly one representation.
    std::vector<int> build;for(int i=0;i<30;++i)if(i==0||i>=5)build.push_back(i);if(order)std::reverse(build.begin(),build.end());
    for(int i:build){auto& c=candidates[i];
        if(i==0){make_mixed(mixed,ptr,col,val,cols);c.slots=mixed.tiles*32;c.bytes=mixed.stored_bytes;}
        else {int tile=i==5?-1:(i>=18?(i-18)/2:(i-6)/2),bal=i==5?-1:(i>=18?1:(i-6)%2);
            const char* previous=std::getenv("ATSPMV_TUNE_CAP");std::string saved=previous?previous:"";bool had_previous=previous!=nullptr;
            if(i>=18)setenv("ATSPMV_TUNE_CAP",(i-18)%2?"16":"4",1);
            try { CSRToAtBSRProduct(a,c.a,tile,bal); } catch (...) { if(had_previous)setenv("ATSPMV_TUNE_CAP",saved.c_str(),1);else unsetenv("ATSPMV_TUNE_CAP"); throw; }
            if(had_previous)setenv("ATSPMV_TUNE_CAP",saved.c_str(),1);else unsetenv("ATSPMV_TUNE_CAP");c.slots=c.a.blockCount*32;
            c.bytes=size_t(c.slots)*(sizeof(double)+(c.a.useColIdxInt16?sizeof(int16_t):sizeof(int32_t)))+size_t(rows)*sizeof(int)+size_t(c.a.brow+1)*sizeof(int)+size_t(c.a.warpCount)*sizeof(int)+size_t(c.a.warpCount+1)*sizeof(int);
        }
    }
    candidates[1].slots=candidates[0].slots;candidates[1].bytes=candidates[0].bytes;
    candidates[2].slots=candidates[3].slots=candidates[4].slots=candidates[0].slots;
    candidates[2].bytes=mixed.bytes4;candidates[3].bytes=mixed.bytes8;candidates[4].bytes=mixed.bytes16;
    for(int i=6;i<18;++i)if(candidates[0].slots>candidates[i].slots)
        throw std::logic_error("mixed row partition uses more slots than a fixed shape");
    auto launch=[&](Candidate& c){
        if (c.mixed) {
            const int caps[] = {0, -1, 0, 4, 8, 16};
            atspmv::detail::launchMixed(mixed, caps[c.mixed], rows, dx, dy, stream);
        } else {
            AtSpmvProduct(c.a, dx, dy, 1, 0, stream);
        }
    };
    if(const char* profile_name=std::getenv("ATSPMV_MIXED_PROFILE")) {
        auto it=std::find_if(candidates.begin(),candidates.end(),[&](const Candidate& c){return c.name==profile_name;});
        if(it==candidates.end())throw std::invalid_argument("unknown profile candidate");
        for(int i=0;i<10;++i)launch(*it);
        ck(cudaStreamSynchronize(stream));ck(cudaProfilerStart());launch(*it);ck(cudaStreamSynchronize(stream));ck(cudaProfilerStop());
        return 0;
    }
    auto verify=[&](const Candidate& c){ck(cudaStreamSynchronize(stream));ck(cudaMemcpy(y.data(),dy,size_t(rows)*sizeof(double),cudaMemcpyDeviceToHost));
        for(int row=0;row<rows;++row){long double sum=0,mag=0;for(int j=ptr[row];j<ptr[row+1];++j){long double p=(long double)val[j]*x[col[j]];sum+=p;mag+=std::abs(p);}
            long double tol=8*std::numeric_limits<double>::epsilon()*(ptr[row+1]-ptr[row]+32.L)*mag+1e-12L;
            if(!std::isfinite(y[row])||std::abs(y[row]-sum)>tol)throw std::runtime_error("incorrect "+c.name+" row="+std::to_string(row));}
    };
    for(auto& c:candidates){
        for(int v=0;v<2;++v){for(int i=0;i<cols;++i)x[i]=((i*(v?31LL:23LL)+7)%113-56)/67.;
            ck(cudaMemcpyAsync(dx,x.data(),size_t(cols)*sizeof(double),cudaMemcpyHostToDevice,stream));
            ck(cudaMemsetAsync(dy,0xff,size_t(rows)*sizeof(double),stream));launch(c);verify(c);
        }
        ck(cudaEventCreate(&c.begin));ck(cudaEventCreate(&c.end));ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        ck(cudaEventRecordWithFlags(c.begin,stream,cudaEventRecordExternal));for(int i=0;i<8;++i)launch(c);
        ck(cudaEventRecordWithFlags(c.end,stream,cudaEventRecordExternal));ck(cudaStreamEndCapture(stream,&c.graph));
        ck(cudaGraphInstantiate(&c.exec,c.graph,nullptr,nullptr,0));ck(cudaMemsetAsync(dy,0xff,size_t(rows)*sizeof(double),stream));
        ck(cudaGraphLaunch(c.exec,stream));verify(c);
    }
    std::vector<int> indices(candidates.size());std::iota(indices.begin(),indices.end(),0);std::mt19937 rng(seed);
    for(int t=-3;t<trials;++t){std::shuffle(indices.begin(),indices.end(),rng);for(int i:indices){auto& c=candidates[i];
        ck(cudaGraphLaunch(c.exec,stream));ck(cudaGraphLaunch(c.exec,stream));ck(cudaStreamSynchronize(stream));float ms=0;ck(cudaEventElapsedTime(&ms,c.begin,c.end));if(t>=0)c.samples.push_back(ms*1000./8.);
    }}
    std::cout<<"candidate,trial,us,slots,stored_bytes,rows,cols,nnz,groups_m1,groups_m2,groups_m4,groups_m8,groups_m16,groups_m32\n"<<std::setprecision(10);
    for(auto& c:candidates)for(int t=0;t<trials;++t){std::cout<<c.name<<','<<t<<','<<c.samples[t]<<','<<c.slots<<','<<c.bytes<<','<<rows<<','<<cols<<','<<nnz;for(int n:mixed.histogram)std::cout<<','<<n;std::cout<<'\n';}
    for(auto& c:candidates){ck(cudaGraphExecDestroy(c.exec));ck(cudaGraphDestroy(c.graph));ck(cudaEventDestroy(c.begin));ck(cudaEventDestroy(c.end));if(!c.mixed)free_atbsr(c.a);}
    ck(cudaFree(a.row_ptr));ck(cudaFree(a.col_idx));ck(cudaFree(a.val));ck(cudaFree(dx));ck(cudaFree(dy));ck(cudaStreamDestroy(stream));return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
