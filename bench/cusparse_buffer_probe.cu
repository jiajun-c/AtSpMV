// Isolate the effect of distinct CSR, X and Y allocations on cuSPARSE timings.
#include "csrReader.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void cs(cusparseStatus_t s) { if(s!=CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cuSPARSE status "+std::to_string(int(s))); }
template<class T> T* upload(const std::vector<T>& v) {
    T* p=nullptr;ck(cudaMalloc(&p,std::max(size_t(1),v.size())*sizeof(T)));
    if(!v.empty()) ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));
    return p;
}
struct Csr { int* ptr;int* col;double* val; };
struct Trial {
    int matrix,x,y;
    cusparseHandle_t handle{};cusparseSpMatDescr_t mat{};cusparseDnVecDescr_t vx{},vy{};
    void* workspace{};cudaGraph_t graph{};cudaGraphExec_t exec{};
    std::vector<double> samples;
};
int main(int argc,char** argv) { try {
    if(argc!=4) throw std::invalid_argument("usage: cusparse_buffer_probe matrix.mtx seed trials");
    unsigned seed=std::stoul(argv[2]);int trials=std::stoi(argv[3]);
    if(trials<1)throw std::invalid_argument("trials must be positive");
    auto a=read_mtx(argv[1]);const int m=a.nrows,n=a.ncols,nnz=a.nnz;
    std::vector<int> ptr(a.row_ptr,a.row_ptr+m+1),col(a.col_idx,a.col_idx+nnz);
    std::vector<double> values(a.val,a.val+nnz),host_x(n),result(m),reference(m);
    delete[] a.row_ptr;delete[] a.col_idx;delete[] a.val;
    for(int i=0;i<n;++i)host_x[i]=((i*23LL)%97-48)/61.;
    for(int row=0;row<m;++row) { long double sum=0;for(int j=ptr[row];j<ptr[row+1];++j)sum+=(long double)values[j]*host_x[col[j]];reference[row]=double(sum); }
    Csr matrices[2];for(auto& d:matrices){d.ptr=upload(ptr);d.col=upload(col);d.val=upload(values);}
    double *xs[2],*ys[2];for(auto& x:xs)x=upload(host_x);for(auto& y:ys)y=upload(result);
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    double alpha=1,beta=0;std::vector<Trial> cases(8);int k=0;
    for(int matrix=0;matrix<2;++matrix)for(int x=0;x<2;++x)for(int y=0;y<2;++y) {
        auto& t=cases[k++];t.matrix=matrix;t.x=x;t.y=y;
        cs(cusparseCreate(&t.handle));cs(cusparseSetStream(t.handle,stream));
        cs(cusparseCreateCsr(&t.mat,m,n,nnz,matrices[matrix].ptr,matrices[matrix].col,matrices[matrix].val,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
        cs(cusparseCreateDnVec(&t.vx,n,xs[x],CUDA_R_64F));cs(cusparseCreateDnVec(&t.vy,m,ys[y],CUDA_R_64F));
        size_t bytes=0;cs(cusparseSpMV_bufferSize(t.handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,t.mat,t.vx,&beta,t.vy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,&bytes));
        ck(cudaMalloc(&t.workspace,std::max(size_t(1),bytes)));
        cs(cusparseSpMV_preprocess(t.handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,t.mat,t.vx,&beta,t.vy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,t.workspace));
        auto launch=[&]{cs(cusparseSpMV(t.handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,t.mat,t.vx,&beta,t.vy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,t.workspace));};
        launch();ck(cudaStreamSynchronize(stream));ck(cudaMemcpy(result.data(),ys[y],m*sizeof(double),cudaMemcpyDeviceToHost));
        for(int row=0;row<m;++row)if(!std::isfinite(result[row])||std::abs(result[row]-reference[row])>1e-9*(1+std::abs(reference[row])))throw std::runtime_error("result mismatch at row "+std::to_string(row));
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));for(int i=0;i<8;++i)launch();
        ck(cudaStreamEndCapture(stream,&t.graph));ck(cudaGraphInstantiate(&t.exec,t.graph,nullptr,nullptr,0));
        ck(cudaMemsetAsync(ys[y],0xff,size_t(m)*sizeof(double),stream));ck(cudaGraphLaunch(t.exec,stream));ck(cudaStreamSynchronize(stream));
        ck(cudaMemcpy(result.data(),ys[y],m*sizeof(double),cudaMemcpyDeviceToHost));
        for(int row=0;row<m;++row)if(!std::isfinite(result[row])||std::abs(result[row]-reference[row])>1e-9*(1+std::abs(reference[row])))throw std::runtime_error("graph mismatch at row "+std::to_string(row));
    }
    cudaEvent_t start,stop;ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));
    std::mt19937 random(seed);std::vector<int> order(8);std::iota(order.begin(),order.end(),0);
    for(int batch=-3;batch<trials;++batch){std::shuffle(order.begin(),order.end(),random);for(int i:order){auto& t=cases[i];
        ck(cudaGraphLaunch(t.exec,stream));ck(cudaEventRecord(start,stream));ck(cudaGraphLaunch(t.exec,stream));
        ck(cudaEventRecord(stop,stream));ck(cudaEventSynchronize(stop));float ms;ck(cudaEventElapsedTime(&ms,start,stop));
        if(batch>=0)t.samples.push_back(ms*1000/8);
    }}
    std::cout<<"matrix_buffer,x_buffer,y_buffer,trial,us\n"<<std::setprecision(10);
    for(auto& t:cases)for(int i=0;i<trials;++i)std::cout<<t.matrix<<','<<t.x<<','<<t.y<<','<<i<<','<<t.samples[i]<<'\n';
    for(auto& t:cases){ck(cudaGraphExecDestroy(t.exec));ck(cudaGraphDestroy(t.graph));cs(cusparseDestroyDnVec(t.vx));cs(cusparseDestroyDnVec(t.vy));cs(cusparseDestroySpMat(t.mat));cs(cusparseDestroy(t.handle));ck(cudaFree(t.workspace));}
    for(auto& d:matrices){ck(cudaFree(d.ptr));ck(cudaFree(d.col));ck(cudaFree(d.val));}for(auto x:xs)ck(cudaFree(x));for(auto y:ys)ck(cudaFree(y));
    ck(cudaEventDestroy(start));ck(cudaEventDestroy(stop));ck(cudaStreamDestroy(stream));
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
