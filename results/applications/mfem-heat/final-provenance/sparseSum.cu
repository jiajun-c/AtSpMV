#include "sparseSum.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

namespace atspmv_app {
namespace {
void Check(cudaError_t status) {
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
__global__ void SumValues(int n, const int* map_a, const int* map_b,
    const double* a, const double* b, double weight, double* out) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p<n) {
        double av=map_a[p]>=0?a[map_a[p]]:0.;
        double bv=map_b[p]>=0?b[map_b[p]]:0.;
        out[p]=av+weight*bv;
    }
}
}
void SparseSum::ReleaseMaps() {
    cudaFree(map_a_);cudaFree(map_b_);map_a_=map_b_=nullptr;
}
SparseSum::~SparseSum() { ReleaseMaps(); }
mfem::SparseMatrix* SparseSum::Update(const mfem::SparseMatrix& a,
    const mfem::SparseMatrix& b, double weight, bool reuse) {
    const int *ap=a.HostReadI(), *aj=a.HostReadJ(), *bp=b.HostReadI(), *bj=b.HostReadJ();
    reused=reuse && map_a_ && map_b_ && matrix_ && matrix_->Height()==a.Height() && matrix_->Width()==a.Width() &&
        ap_.size()==size_t(a.Height()+1) && bp_.size()==size_t(b.Height()+1) &&
        aj_.size()==size_t(a.NumNonZeroElems()) && bj_.size()==size_t(b.NumNonZeroElems()) &&
        std::equal(ap_.begin(),ap_.end(),ap) && std::equal(aj_.begin(),aj_.end(),aj) &&
        std::equal(bp_.begin(),bp_.end(),bp) && std::equal(bj_.begin(),bj_.end(),bj);
    if(reused) {
        int n=matrix_->NumNonZeroElems();
        if(n)SumValues<<<(n+255)/256,256>>>(n,map_a_,map_b_,a.ReadData(),b.ReadData(),weight,matrix_->WriteData());
        Check(cudaGetLastError());
        return matrix_.get();
    }
    ReleaseMaps();
    matrix_.reset(mfem::Add(1.,a,weight,b));
    if(!reuse)return matrix_.get();
    ap_.assign(ap,ap+a.Height()+1);aj_.assign(aj,aj+a.NumNonZeroElems());
    bp_.assign(bp,bp+b.Height()+1);bj_.assign(bj,bj+b.NumNonZeroElems());
    const int *cp=matrix_->HostReadI(), *cj=matrix_->HostReadJ();
    std::vector<int> ma(matrix_->NumNonZeroElems(),-1),mb(ma.size(),-1);
    std::vector<int> marker_a(a.Width(),-1),marker_b(b.Width(),-1);
    for(int row=0;row<a.Height();++row) {
        for(int p=ap[row];p<ap[row+1];++p)marker_a[aj[p]]=p;
        for(int p=bp[row];p<bp[row+1];++p)marker_b[bj[p]]=p;
        for(int p=cp[row];p<cp[row+1];++p) {
            int ia=marker_a[cj[p]], ib=marker_b[cj[p]];
            if(ia>=ap[row] && ia<ap[row+1])ma[p]=ia;
            if(ib>=bp[row] && ib<bp[row+1])mb[p]=ib;
        }
    }
    size_t bytes=std::max(size_t(1),ma.size())*sizeof(int);
    Check(cudaMalloc(&map_a_,bytes));Check(cudaMalloc(&map_b_,bytes));
    if(!ma.empty()) {
        Check(cudaMemcpy(map_a_,ma.data(),ma.size()*sizeof(int),cudaMemcpyHostToDevice));
        Check(cudaMemcpy(map_b_,mb.data(),mb.size()*sizeof(int),cudaMemcpyHostToDevice));
    }
    return matrix_.get();
}
} // namespace atspmv_app
