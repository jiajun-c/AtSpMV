#include "spmvPlan.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void check(cudaError_t status) {
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> T* upload(const std::vector<T>& data) {
    T* p=nullptr;check(cudaMalloc(&p,std::max(size_t(1),data.size())*sizeof(T)));
    if(!data.empty())check(cudaMemcpy(p,data.data(),data.size()*sizeof(T),cudaMemcpyHostToDevice));
    return p;
}
int main(int argc,char** argv) {try {
    if(argc>2||(argc==2&&std::string(argv[1])!="--compressed-only"))
        throw std::invalid_argument("usage: spmv_value_update_test [--compressed-only]");
    const bool compressed_only=argc==2;
    for(int columns:{83,65536,200000}) {
    const int n=columns==200000?129:83;
    std::vector<int> ptr{0},col;
    std::vector<double> initial, updated;
    for(int row=0;row<n;++row) {
        int count=row%11==0?73:row%17;
        if(columns==200000)count=row<32?0:row==128?5:7;
        for(int j=0;j<count;++j) {
            int column=(j*19+row*7)%columns;
            if(columns==200000)column=row<64||row==128?1000+j+(row%32):
                row<96?10000+j*7000+(row%32):j*30000+(row%32);
            col.push_back(column);
            initial.push_back((j+1.)/(row+2.));
            updated.push_back((j%3==0?0.:-1.)*(row+1.)/(j+2.));
        }
        ptr.push_back(col.size());
    }
    std::vector<double> x(columns),out(n);
    for(int i=0;i<columns;++i)x[i]=(i%13-6)/7.;
    CSRMatrix<double,int32_t> csr;
    csr.nrows=n;csr.ncols=columns;csr.nnz=initial.size();
    csr.row_ptr=upload(ptr);csr.col_idx=upload(col);csr.val=upload(initial);
    auto* next=upload(updated);auto* dx=upload(x);auto* dy=upload(out);
    cudaStream_t stream;check(cudaStreamCreate(&stream));
    auto verify=[&](const std::vector<double>& values) {
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(out.data(),dy,n*sizeof(double),cudaMemcpyDeviceToHost));
        for(int row=0;row<n;++row) {
            long double ref=0,mag=0;
            for(int p=ptr[row];p<ptr[row+1];++p) {long double term=(long double)values[p]*x[col[p]];ref+=term;mag+=std::abs(term);}
            if(!std::isfinite(out[row])||std::abs(out[row]-ref)>1e-11*(1+mag))
                throw std::runtime_error("updated values were not observed");
        }
    };
    int configs=0;
    auto run=[&](atspmv::SpmvPlanOptions options) {
        if(compressed_only&&!options.compress_tile_columns)return;
        options.enable_value_updates=true;
        auto plan=atspmv::makeSpmvPlan(csr,options,stream);
        if(!plan.info().value_updates_enabled)throw std::runtime_error("updates unavailable");
        if(options.compress_tile_columns && (columns>=65536)!=plan.info().tile_columns_compressed)
            throw std::runtime_error("unexpected compressed layout availability");
        atspmv::executeSpmv(plan,dx,dy,stream);verify(initial);
        atspmv::updateSpmvValues(plan,next,stream);
        atspmv::executeSpmv(plan,dx,dy,stream);verify(updated);
        cudaGraph_t graph;cudaGraphExec_t exec;
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        atspmv::updateSpmvValues(plan,csr.val,stream);
        atspmv::executeSpmv(plan,dx,dy,stream);
        check(cudaStreamEndCapture(stream,&graph));
        check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        check(cudaGraphLaunch(exec,stream));verify(initial);
        // A second update before replay verifies ordering and independence
        // from the previous output, including padding and balanced atomics.
        atspmv::updateSpmvValues(plan,next,stream);
        check(cudaGraphLaunch(exec,stream));verify(initial);
        check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));
        bool rejected=false;
        try {atspmv::updateSpmvValues(plan,nullptr,stream);}catch(const std::invalid_argument&) {rejected=true;}
        if(!rejected)throw std::runtime_error("null values accepted");
        ++configs;
    };
    for(int tile=0;tile<6;++tile)for(int balance=0;balance<2;++balance) {
        atspmv::SpmvPlanOptions o;o.backend=atspmv::SpmvBackend::AtBsr;o.tile_index=tile;o.balanced=balance;run(o);
    }
    {atspmv::SpmvPlanOptions o;o.backend=atspmv::SpmvBackend::AtBsr;
     o.tile_index=5;o.balanced=0;o.compress_tile_columns=true;run(o);}
    for(int cap:{-1,0,4,8,16}) {
        atspmv::SpmvPlanOptions o;o.backend=atspmv::SpmvBackend::MixedTile;o.mixed_tile_cap=cap;run(o);
    }
    for(int width:{1,2,4,8,16,32}) {
        atspmv::SpmvPlanOptions o;o.backend=atspmv::SpmvBackend::CsrVector;o.vector_width=width;run(o);
    }
    for(auto backend:{atspmv::SpmvBackend::CuSparse,atspmv::SpmvBackend::CsrTwoStage}) {
        atspmv::SpmvPlanOptions o;o.backend=backend;o.long_rows.threshold=0;run(o);
    }
    {atspmv::SpmvPlanOptions o;auto plan=atspmv::makeSpmvPlan(csr,o,stream);
     if(plan.info().dsm_available)for(auto backend:{atspmv::SpmvBackend::CsrDsm,atspmv::SpmvBackend::CsrDsmVector}) {
         atspmv::SpmvPlanOptions d;d.backend=backend;d.long_rows={2,128,1,0};run(d);
     }
     bool rejected=false;try{atspmv::updateSpmvValues(plan,next,stream);}catch(const std::invalid_argument&){rejected=true;}
     if(!rejected)throw std::runtime_error("disabled update accepted");}
    check(cudaStreamDestroy(stream));
    for(void* p:{(void*)csr.row_ptr,(void*)csr.col_idx,(void*)csr.val,(void*)next,(void*)dx,(void*)dy})check(cudaFree(p));
    std::cout<<"value update configurations="<<configs<<" correct=1\n";
    }
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}}
