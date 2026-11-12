// Repeated, counterbalanced Plan construction measurements. Matrix loading,
// factorization, Graph capture and solve are deliberately outside the timer.
#include "sptrsvPlan.h"
#include "csrReader.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
using Clock=std::chrono::steady_clock;
int main(int argc,char**argv){try{
    if(argc!=2 && !(argc==3 && std::string(argv[2])=="--reverse"))throw std::invalid_argument("usage: sptrsv_plan_setup_bench factor.mtx [--reverse]");
    auto src=read_mtx(argv[1]);if(src.nrows<=0 || src.nrows!=src.ncols)throw std::runtime_error("nonempty square factor required");
    atspmv::SptrsvHostCsr a;a.rows=src.nrows;a.row_ptr.assign(src.row_ptr,src.row_ptr+a.rows+1);a.col_idx.assign(src.col_idx,src.col_idx+src.nnz);a.values.assign(src.val,src.val+src.nnz);
    delete[] src.row_ptr;delete[] src.col_idx;delete[] src.val;
    double *b,*x;check(cudaMalloc(&b,a.rows*sizeof(double)));check(cudaMalloc(&x,a.rows*sizeof(double)));check(cudaMemset(b,0,a.rows*sizeof(double)));check(cudaMemset(x,0,a.rows*sizeof(double)));
    cudaStream_t stream;check(cudaStreamCreate(&stream));
    for(int mode=0;mode<2;++mode){atspmv::SptrsvPlanOptions o;o.enable_cached_tile_model=mode;auto p=atspmv::makeSptrsvPlan(a,b,x,o,stream);atspmv::executeSptrsv(p,stream);check(cudaStreamSynchronize(stream));}
    std::vector<double> setup[2],destroy[2];atspmv::SptrsvBackend backend[2]{};
    for(int iteration=0;iteration<12;++iteration)for(int step=0;step<2;++step) {
        int mode=(step+(iteration%2)+(argc==3))%2;atspmv::SptrsvPlanOptions options;options.enable_cached_tile_model=mode;
        auto start=Clock::now();auto plan=atspmv::makeSptrsvPlan(a,b,x,options,stream);auto ready=Clock::now();backend[mode]=plan.info().backend;
        plan=atspmv::SptrsvPlan{};auto released=Clock::now();
        setup[mode].push_back(std::chrono::duration<double,std::micro>(ready-start).count());
        destroy[mode].push_back(std::chrono::duration<double,std::micro>(released-ready).count());
    }
    std::cout<<std::setprecision(10)<<"matrix,mode,backend,setup_median_us,setup_min_us,setup_max_us,destroy_median_us,iterations\n";
    for(int mode=0;mode<2;++mode){std::sort(setup[mode].begin(),setup[mode].end());std::sort(destroy[mode].begin(),destroy[mode].end());
        const char* name=backend[mode]==atspmv::SptrsvBackend::CachedTile?"cached":backend[mode]==atspmv::SptrsvBackend::CachedCsr?"cached-csr":"cusparse";
        auto median=[](const std::vector<double>& x){return (x[5]+x[6])/2.;};
        std::cout<<argv[1]<<','<<(mode?"model":"default")<<','<<name<<','<<median(setup[mode])<<','<<setup[mode].front()<<','<<setup[mode].back()<<','<<median(destroy[mode])<<','<<setup[mode].size()<<'\n';
    }
    check(cudaStreamDestroy(stream));check(cudaFree(b));check(cudaFree(x));return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
