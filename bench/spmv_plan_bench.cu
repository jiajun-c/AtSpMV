#include "spmvPlan.h"
#include "csrReader.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>T* upload(const std::vector<T>& v){T* p;check(cudaMalloc(&p,std::max(size_t(1),v.size())*sizeof(T)));if(!v.empty())check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
int main(int argc,char** argv){try{
    if(argc<2||argc>3)throw std::invalid_argument("Usage: spmv_plan_bench MATRIX.mtx|--synthetic [--tune|--tune-mixed|--fast-tune-mixed|--mixed|--model-refine|--model-refine-mixed]");
    atspmv::SpmvPlanOptions options;
    if(argc==3) {
        const std::string mode=argv[2];
        if(mode=="--tune")options.tune=true;
        else if(mode=="--tune-mixed"){options.tune=true;options.enable_mixed_tile=true;}
        else if(mode=="--fast-tune-mixed"){options.tune=true;options.enable_mixed_tile=true;options.fast_tune=true;}
        else if(mode=="--mixed")options.backend=atspmv::SpmvBackend::MixedTile;
        else if(mode=="--model-refine"||mode=="--model-refine-mixed") {
            options.model_select=true;options.vector_model_select=true;options.model_refine=true;
            options.enable_mixed_tile=(mode=="--model-refine-mixed");
        }
        else throw std::invalid_argument("unknown option");
    }
    std::vector<int> ptr,col;std::vector<double> val;int m,n;
    if(std::string(argv[1])=="--synthetic") {
        m=64;n=65536;ptr.push_back(0);
        for(int r=0;r<m;++r){for(int j=0;j<16384;++j){col.push_back((j*4+r*17)%n);val.push_back(((j*17LL+r*7)%127-63)/79.);}ptr.push_back(col.size());}
    }else{
        auto a=read_mtx(argv[1]);m=a.nrows;n=a.ncols;
        ptr.assign(a.row_ptr,a.row_ptr+m+1);col.assign(a.col_idx,a.col_idx+a.nnz);val.assign(a.val,a.val+a.nnz);
        delete[]a.row_ptr;delete[]a.col_idx;delete[]a.val;
    }
    CSRMatrix<double,int32_t> input;input.nrows=m;input.ncols=n;input.nnz=val.size();
    input.row_ptr=upload(ptr);input.col_idx=upload(col);input.val=upload(val);
    auto plan=atspmv::makeSpmvPlan(input,options);
    check(cudaFree(input.row_ptr));check(cudaFree(input.col_idx));check(cudaFree(input.val));
    // Input ownership ends here; only the plan is needed for execution.
    std::vector<double> x(n),y(m);for(int j=0;j<n;++j)x[j]=((j*31LL+7)%113-56)/67.;
    double* dx=upload(x);double* dy=upload(y);
    atspmv::executeSpmv(plan,dx,dy);check(cudaDeviceSynchronize());
    check(cudaMemcpy(y.data(),dy,m*sizeof(double),cudaMemcpyDeviceToHost));
    for(int r=0;r<m;++r){long double ref=0,mag=0;for(int j=ptr[r];j<ptr[r+1];++j){long double v=(long double)val[j]*x[col[j]];ref+=v;mag+=std::abs(v);}
        if(!std::isfinite(y[r])||std::abs(y[r]-ref)>8*std::numeric_limits<double>::epsilon()*(ptr[r+1]-ptr[r]+32.L)*mag+1e-12L)throw std::runtime_error("incorrect result");}
    const auto& info=plan.info();
    std::cout<<std::setprecision(9)<<"selected="<<atspmv::spmvBackendName(info.backend)<<", measured="<<info.measured<<", setup_us="<<info.setup_us<<", model_refined="<<info.model_refined<<", dsm_available="<<info.dsm_available<<", vector_width="<<info.vector_width<<", mixed_tile_cap="<<info.mixed_tile_cap<<", correct=1\n";
    std::cout<<"backend,tile_index,balanced,parts,threads,rows_per_group,threshold,median_us,vector_width,mixed_tile_cap\n";
    for(auto& c:info.timings)std::cout<<atspmv::spmvBackendName(c.backend)<<','<<c.tile_index<<','<<c.balanced<<','<<c.long_rows.parts<<','<<c.long_rows.threads<<','<<c.long_rows.rows_per_group<<','<<c.long_rows.threshold<<','<<c.median_us<<','<<c.vector_width<<','<<c.mixed_tile_cap<<'\n';
    check(cudaFree(dx));check(cudaFree(dy));
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
