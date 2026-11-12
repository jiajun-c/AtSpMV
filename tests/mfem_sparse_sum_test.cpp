#include "sparseSum.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main() {try {
    mfem::Device device("cuda");
    const int n=4;
    mfem::SparseMatrix a(n),b(n);
    for(int row=0;row<n;++row) {
        a.Add(row,row,2.+row);
        a.Add(row,(row+1)%n,-.5);
        b.Add(row,(row+2)%n,1.+row);
        b.Add(row,row,.25);
    }
    a.Finalize();b.Finalize();
    std::vector<double> av(a.HostReadData(),a.HostReadData()+a.NumNonZeroElems());
    std::vector<double> bv(b.HostReadData(),b.HostReadData()+b.NumNonZeroElems());
    std::vector<int> ap(a.HostReadI(),a.HostReadI()+n+1), aj(a.HostReadJ(),a.HostReadJ()+a.NumNonZeroElems());
    std::vector<int> bp(b.HostReadI(),b.HostReadI()+n+1), bj(b.HostReadJ(),b.HostReadJ()+b.NumNonZeroElems());
    mfem::Vector x(n),y(n);for(int i=0;i<n;++i)x[i]=.3*(i+1);
    atspmv_app::SparseSum cache;
    auto run=[&](double weight,bool reuse,bool expected_reuse) {
        auto* sum=cache.Update(a,b,weight,reuse);
        if(cache.reused!=expected_reuse)throw std::runtime_error("incorrect reuse decision");
        sum->Mult(x,y);check(cudaDeviceSynchronize());
        const double* out=y.HostRead();
        for(int row=0;row<n;++row) {
            double ref=0;
            for(int p=ap[row];p<ap[row+1];++p)ref+=av[p]*x[aj[p]];
            for(int p=bp[row];p<bp[row+1];++p)ref+=weight*bv[p]*x[bj[p]];
            if(std::abs(out[row]-ref)>1e-12)throw std::runtime_error("incorrect sparse sum");
        }
    };
    run(.5,true,false);
    for(size_t i=0;i<av.size();++i)av[i]=i%2?0.:-av[i];
    for(double& v:bv)v*=3.;
    check(cudaMemcpy(a.WriteData(),av.data(),av.size()*sizeof(double),cudaMemcpyHostToDevice));
    check(cudaMemcpy(b.WriteData(),bv.data(),bv.size()*sizeof(double),cudaMemcpyHostToDevice));
    run(.25,true,true);run(-.75,true,true);
    run(.5,false,false);run(.5,true,false);run(.5,true,true);
    // Same dimensions and entry count, different entry ordering: rebuild maps.
    auto* cols=b.HostReadWriteJ();auto* values=b.HostReadWriteData();
    std::swap(cols[0],cols[1]);std::swap(values[0],values[1]);
    std::swap(bj[0],bj[1]);std::swap(bv[0],bv[1]);
    b.ClearGPUSparse();
    run(.1,true,false);run(.2,true,true);
    std::cout<<"sparse sum: initial, numerical update, changed weight, disabled reuse, structural change passed\n";
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
