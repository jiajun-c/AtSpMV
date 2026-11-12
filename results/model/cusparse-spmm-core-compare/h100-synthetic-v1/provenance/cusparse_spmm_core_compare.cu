// Compare cuSPARSE CSR and Blocked-ELL on identical synthetic matrices.
// FP16 inputs, FP32 output/accumulation; excludes host format construction.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void ck(cusparseStatus_t e) { if (e != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error(cusparseGetErrorString(e)); }
template<class T> static T* upload(const std::vector<T>& x) {
    T* p = nullptr; ck(cudaMalloc(&p, std::max<size_t>(1, x.size()) * sizeof(T)));
    if (!x.empty()) ck(cudaMemcpy(p, x.data(), x.size()*sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
struct Gpu {
    void* p = nullptr;
    explicit Gpu(size_t n) { ck(cudaMalloc(&p, std::max<size_t>(n, 1))); }
    ~Gpu() { cudaFree(p); }
};

int main(int argc, char** argv) {
 try {
    const int m = 1024, k = 1024, block = 16;
    const int blocks_per_row = argc > 1 ? std::atoi(argv[1]) : 4;
    const int percent = argc > 2 ? std::atoi(argv[2]) : 25;
    const int n = argc > 3 ? std::atoi(argv[3]) : 32;
    const std::string only = argc > 4 ? argv[4] : "all";
    const bool reverse = argc > 5 && std::string(argv[5]) == "reverse";
    if (blocks_per_row < 1 || blocks_per_row > k/block || percent < 1 || percent > 100 || n < 1) return 2;
    std::mt19937 rng(20261102);
    const int ellcols = blocks_per_row * block;
    std::vector<int> ellidx(m/block * blocks_per_row), rowptr(m+1), col;
    std::vector<__half> ellval(size_t(m)*ellcols, __float2half(0.f)), val;
    for (int br = 0; br < m/block; ++br) {
        std::vector<int> columns(k/block);
        for (int i=0;i<k/block;++i) columns[i]=i;
        std::shuffle(columns.begin(), columns.end(), rng);
        columns.resize(blocks_per_row); std::sort(columns.begin(), columns.end());
        for (int j=0;j<blocks_per_row;++j) ellidx[br*blocks_per_row+j]=columns[j];
        for (int r=0;r<block;++r) {
            const int row=br*block+r;
            for (int j=0;j<blocks_per_row;++j) for (int c=0;c<block;++c) {
                if (int(rng()%100) >= percent) continue;
                const float value = float(int(rng()%15)-7)/8.0f;
                if (value == 0) continue;
                const __half a = __float2half(value);
                ellval[size_t(row)*ellcols+j*block+c] = a;
                col.push_back(columns[j]*block+c); val.push_back(a);
            }
            rowptr[row+1]=int(col.size());
        }
    }
    // Independently reconstruct each CSR row from the ELL slots, including
    // zeros, to detect representation errors before any GPU comparison.
    for (int row=0;row<m;++row) {
        int q=rowptr[row];
        for (int j=0;j<blocks_per_row;++j) for (int c=0;c<block;++c) {
            const __half a=ellval[size_t(row)*ellcols+j*block+c];
            if (__half2float(a)==0.f) continue;
            const int index=ellidx[(row/block)*blocks_per_row+j]*block+c;
            if (q>=rowptr[row+1] || col[q]!=index || __half2float(val[q])!=__half2float(a))
                throw std::runtime_error("CSR/Blocked-ELL mismatch");
            ++q;
        }
        if(q!=rowptr[row+1]) throw std::runtime_error("CSR/Blocked-ELL row length mismatch");
    }
    if (std::getenv("ATSPMV_SPMM_HOST_CHECK")) {
        std::printf("host-only PASS: nnz=%zu blocked_slots=%zu\n",val.size(),ellval.size());
        return 0;
    }
    std::vector<__half> B(size_t(k)*n);
    for (auto& x:B) x=__float2half(float(int(rng()%15)-7)/8.f);
    std::vector<float> reference(size_t(m)*n,0.f);
    for(int r=0;r<m;++r) for(int q=rowptr[r];q<rowptr[r+1];++q)
        for(int j=0;j<n;++j) reference[size_t(r)*n+j] += __half2float(val[q])*__half2float(B[size_t(col[q])*n+j]);
    int* drow=upload(rowptr); int* dcol=upload(col); int* dellidx=upload(ellidx);
    __half* dval=upload(val); __half* dellval=upload(ellval); __half* db=upload(B);
    Gpu output(size_t(m)*n*sizeof(float));
    cusparseHandle_t handle; ck(cusparseCreate(&handle));
    cusparseSpMatDescr_t csr,ell; cusparseDnMatDescr_t b,c;
    ck(cusparseCreateCsr(&csr,m,k,val.size(),drow,dcol,dval,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_16F));
    ck(cusparseCreateBlockedEll(&ell,m,k,block,ellcols,dellidx,dellval,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_16F));
    ck(cusparseCreateDnMat(&b,k,n,n,db,CUDA_R_16F,CUSPARSE_ORDER_ROW));
    ck(cusparseCreateDnMat(&c,m,n,n,output.p,CUDA_R_32F,CUSPARSE_ORDER_ROW));
    const float alpha=1.f,beta=0.f;
    struct Candidate { const char* name; cusparseSpMatDescr_t mat; cusparseSpMMAlg_t algo; };
    Candidate candidates[]={{"csr_alg2",csr,CUSPARSE_SPMM_CSR_ALG2},{"csr_alg3",csr,CUSPARSE_SPMM_CSR_ALG3},{"blocked_ell",ell,CUSPARSE_SPMM_BLOCKED_ELL_ALG1}};
    if (reverse) std::reverse(std::begin(candidates),std::end(candidates));
    printf("method,m,k,n,nnz,block,blocks_per_row,requested_density,real_density,workspace_bytes,median_us,max_abs_error,status\n");
    for (auto cand:candidates) {
        if (only!="all" && only!=cand.name) continue;
        size_t bytes=0;
        auto status=cusparseSpMM_bufferSize(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,cand.mat,b,&beta,c,CUDA_R_32F,cand.algo,&bytes);
        if(status!=CUSPARSE_STATUS_SUCCESS) { printf("%s,%d,%d,%d,%zu,%d,%d,%d,%.8f,0,0,0,buffer_error_%d\n",cand.name,m,k,n,val.size(),block,blocks_per_row,percent,double(val.size())/ellval.size(),int(status));continue; }
        Gpu work(bytes);
        auto call=[&] { return cusparseSpMM(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,cand.mat,b,&beta,c,CUDA_R_32F,cand.algo,work.p); };
        auto pre=cusparseSpMM_preprocess(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,cand.mat,b,&beta,c,CUDA_R_32F,cand.algo,work.p);
        if(pre!=CUSPARSE_STATUS_SUCCESS) { printf("%s,%d,%d,%d,%zu,%d,%d,%d,%.8f,%zu,0,0,preprocess_error_%d\n",cand.name,m,k,n,val.size(),block,blocks_per_row,percent,double(val.size())/ellval.size(),bytes,int(pre));continue; }
        status=call(); if(status!=CUSPARSE_STATUS_SUCCESS) { printf("%s,%d,%d,%d,%zu,%d,%d,%d,%.8f,%zu,0,0,call_error_%d\n",cand.name,m,k,n,val.size(),block,blocks_per_row,percent,double(val.size())/ellval.size(),bytes,int(status));continue; }
        ck(cudaDeviceSynchronize()); std::vector<float> actual(size_t(m)*n);ck(cudaMemcpy(actual.data(),output.p,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
        float maxerr=0.f; bool good=true;
        for(size_t i=0;i<actual.size();++i) { const float e=std::abs(actual[i]-reference[i]); maxerr=std::max(maxerr,e);if(!std::isfinite(actual[i]) || e>0.002f+0.002f*std::abs(reference[i])) good=false; }
        for(int i=0;i<10;++i)ck(call());
        cudaStream_t stream; ck(cudaStreamCreate(&stream));ck(cusparseSetStream(handle,stream));
        cudaGraph_t graph;cudaGraphExec_t executable;
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        constexpr int calls_per_graph=8;
        for(int i=0;i<calls_per_graph;++i)ck(call());
        ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        cudaEvent_t start,stop; ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));
        std::vector<float> samples;constexpr int repeat=32;
        for(int trial=0;trial<9;++trial) { ck(cudaEventRecord(start,stream));for(int i=0;i<repeat;++i)ck(cudaGraphLaunch(executable,stream));ck(cudaEventRecord(stop,stream));ck(cudaEventSynchronize(stop));float ms=0;ck(cudaEventElapsedTime(&ms,start,stop));samples.push_back(ms*1000.f/(repeat*calls_per_graph)); }
        std::sort(samples.begin(),samples.end());
        printf("%s,%d,%d,%d,%zu,%d,%d,%d,%.8f,%zu,%.6f,%.8g,%s\n",cand.name,m,k,n,val.size(),block,blocks_per_row,percent,double(val.size())/ellval.size(),bytes,samples[4],maxerr,good?"ok":"incorrect");
        ck(cudaEventDestroy(start));ck(cudaEventDestroy(stop));ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));ck(cusparseSetStream(handle,nullptr));ck(cudaStreamDestroy(stream));
    }
    ck(cusparseDestroyDnMat(b));ck(cusparseDestroyDnMat(c));ck(cusparseDestroySpMat(csr));ck(cusparseDestroySpMat(ell));ck(cusparseDestroy(handle));
    cudaFree(drow);cudaFree(dcol);cudaFree(dellidx);cudaFree(dval);cudaFree(dellval);cudaFree(db);
    return 0;
 } catch(const std::exception& e) { fprintf(stderr,"error: %s\n",e.what());return 1; }
}
