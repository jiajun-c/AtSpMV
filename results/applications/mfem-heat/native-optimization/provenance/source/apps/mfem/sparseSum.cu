#include "sparseSum.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

namespace atspmv_app {
namespace {
void Check(cudaError_t status) {
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
__global__ void CompareStructure(int rows, int nnz, const int* ap, const int* aj,
    const int* bp, const int* bj, int* different) {
    bool mismatch=false;
    size_t n=size_t(rows)+1+nnz;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x) {
        if(i<=size_t(rows))mismatch |= ap[i]!=bp[i];
        else {size_t p=i-rows-1;mismatch |= aj[p]!=bj[p];}
    }
    if(__syncthreads_or(mismatch) && threadIdx.x==0)atomicExch(different,1);
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
void CsrStructure::Release() {
    cudaFree(ptr_);cudaFree(col_);cudaFree(different_);
    ptr_=col_=different_=nullptr;
}
CsrStructure::~CsrStructure() { Release(); }
bool CsrStructure::Matches(const mfem::SparseMatrix& matrix) const {
    if(!different_ || rows_!=matrix.Height() || cols_!=matrix.Width() || nnz_!=matrix.NumNonZeroElems())return false;
    Check(cudaMemsetAsync(different_,0,sizeof(int)));
    size_t size=size_t(rows_)+1+nnz_;
    int blocks=std::min(size_t(1024),(size+255)/256);
    CompareStructure<<<blocks,256>>>(rows_,nnz_,ptr_,col_,matrix.ReadI(),matrix.ReadJ(),different_);
    Check(cudaGetLastError());
    int result=0;Check(cudaMemcpyAsync(&result,different_,sizeof(int),cudaMemcpyDeviceToHost));
    Check(cudaStreamSynchronize(nullptr));
    return result==0;
}
void CsrStructure::Remember(const mfem::SparseMatrix& matrix) {
    Release();rows_=matrix.Height();cols_=matrix.Width();nnz_=matrix.NumNonZeroElems();
    Check(cudaMalloc(&ptr_,size_t(rows_+1)*sizeof(int)));
    Check(cudaMalloc(&col_,std::max(size_t(1),size_t(nnz_))*sizeof(int)));
    Check(cudaMalloc(&different_,sizeof(int)));
    Check(cudaMemcpyAsync(ptr_,matrix.ReadI(),size_t(rows_+1)*sizeof(int),cudaMemcpyDeviceToDevice));
    if(nnz_)Check(cudaMemcpyAsync(col_,matrix.ReadJ(),size_t(nnz_)*sizeof(int),cudaMemcpyDeviceToDevice));
}
void SparseSum::ReleaseMaps() {
    cudaFree(map_a_);cudaFree(map_b_);map_a_=map_b_=nullptr;
}
SparseSum::~SparseSum() { ReleaseMaps(); }
mfem::SparseMatrix* SparseSum::Update(const mfem::SparseMatrix& a,
    const mfem::SparseMatrix& b, double weight, bool reuse) {
    reused=reuse && map_a_ && map_b_ && matrix_ && matrix_->Height()==a.Height() && matrix_->Width()==a.Width() &&
        a_structure_.Matches(a) && b_structure_.Matches(b);
    if(reused) {
        int n=matrix_->NumNonZeroElems();
        if(n)SumValues<<<(n+255)/256,256>>>(n,map_a_,map_b_,a.ReadData(),b.ReadData(),weight,matrix_->WriteData());
        Check(cudaGetLastError());
        return matrix_.get();
    }
    ReleaseMaps();
    matrix_.reset(mfem::Add(1.,a,weight,b));
    if(!reuse)return matrix_.get();
    a_structure_.Remember(a);b_structure_.Remember(b);
    const int *ap=a.HostReadI(), *aj=a.HostReadJ(), *bp=b.HostReadI(), *bj=b.HostReadJ();
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
