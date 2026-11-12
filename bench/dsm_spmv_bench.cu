#include "dsmSpmv.h"
#include "csrReader.h"
#include <cusparse.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

static void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void sparse(cusparseStatus_t e) { if (e != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error(cusparseGetErrorString(e)); }
template<class T> T* upload(const T* data, std::size_t count) {
    T* p; check(cudaMalloc(&p, std::max(std::size_t(1), count) * sizeof(T)));
    if (count) check(cudaMemcpy(p, data, count * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
static void release(AtBsr<double, int32_t>& a) {
    for (void* p : {static_cast<void*>(a.blcVal), static_cast<void*>(a.blcIdx), static_cast<void*>(a.blcIdxInt16),
                   static_cast<void*>(a.blcPtr), static_cast<void*>(a.recorder),
                   static_cast<void*>(a.targetRowByWarp), static_cast<void*>(a.warpBlcPtr)}) check(cudaFree(p));
}
struct Candidate {
    std::string kind;
    int tile = -1, balanced = -1, cluster = 0, blocks = 0, warps = 0;
    std::size_t shared = 0;
    double setup_us = 0;
    std::function<void()> launch;
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    std::vector<double> timings;
};
int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 4) throw std::invalid_argument("Usage: dsm_spmv_bench MATRIX.mtx [seed=20261030] [--warp-sweep]");
        for (const char* key : {"ATSPMV_TUNE_CAP", "ATSPMV_TUNE_WARPS", "ATSPMV_TUNE_PREFETCH", "ATSPMV_TUNE_CTA_REDUCE"}) unsetenv(key);
        const unsigned seed = argc >= 3 ? std::stoul(argv[2]) : 20261030;
        const bool warp_sweep = argc == 4 && std::string(argv[3]) == "--warp-sweep";
        if (argc == 4 && !warp_sweep) throw std::invalid_argument("unknown option");
        constexpr int nodes = 64, batches = 11;
        auto host = read_mtx(argv[1]);
        if (!host.nrows || !host.ncols || !host.nnz) throw std::invalid_argument("nonempty matrix required");
        CSRMatrix<double, int32_t> device;
        device.nrows = host.nrows; device.ncols = host.ncols; device.nnz = host.nnz;
        device.row_ptr = upload(host.row_ptr, host.nrows+1);
        device.col_idx = upload(host.col_idx, host.nnz); device.val = upload(host.val, host.nnz);
        std::vector<double> x(host.ncols), output(host.nrows);
        for (int i = 0; i < host.ncols; ++i) x[i] = ((i * 23LL) % 97 - 48) * 0.015625;
        double* dx = upload(x.data(), x.size()); double* dy = upload(output.data(), output.size());
        std::vector<long double> reference(host.nrows), magnitude(host.nrows);
        for (int row=0; row<host.nrows; ++row)
            for (int j=host.row_ptr[row]; j<host.row_ptr[row+1]; ++j) {
                long double v = static_cast<long double>(host.val[j]) * x[host.col_idx[j]];
                reference[row] += v; magnitude[row] += std::abs(v);
            }
        cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::vector<Candidate> candidates;
        // Fixed storage keeps all plan references stable during graph execution.
        std::vector<AtBsr<double,int32_t>> packed(12);
        for (int id=0; id<12; ++id) {
            auto& a=packed[id];
            auto t=std::chrono::steady_clock::now();
            CSRToAtBSRProduct(device,a,id/2,id%2);
            const double packing_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count();
            Candidate c; c.kind="legacy"; c.warps=4; c.tile=id/2; c.balanced=id%2; c.setup_us=packing_us;
            c.launch=[&,id] { AtSpmvProduct(packed[id],dx,dy,1,0,stream); };
            candidates.push_back(c);
            for (int cluster : {1,2,4,8}) {
                for (auto storage : {atspmv::XStorage::Global,atspmv::XStorage::DistributedShared})
                for (bool single_block : {false,true})
                for (int warps : (single_block && warp_sweep ? std::vector<int>{0,8,16,32} : std::vector<int>{0})) {
                    try {
                        t=std::chrono::steady_clock::now();
                        auto plan=atspmv::makeDsmSpmvPlan(a,cluster,storage,0,single_block,warps);
                        Candidate d; d.kind=storage==atspmv::XStorage::Global?"cluster_global":"dsm";
                        if(single_block) d.kind=storage==atspmv::XStorage::Global?"one_cta_global":"one_cta_dsm";
                        d.tile=id/2; d.balanced=id%2; d.cluster=cluster;
                        d.warps=plan.warpsPerBlock(); d.blocks=plan.gridBlocks(); d.shared=plan.sharedBytesPerBlock();
                        d.setup_us=packing_us+std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count();
                        d.launch=[&,plan] { atspmv::launchDsmSpmv(plan,dx,dy,stream); };
                        candidates.push_back(d);
                    } catch (const std::invalid_argument& e) {
                        if (std::string(e.what()).find("entire x does not fit")==std::string::npos &&
                            std::string(e.what())!="DSM block configuration cannot be scheduled") throw;
                    }
                }
            }
        }
        cusparseHandle_t handle; sparse(cusparseCreate(&handle)); sparse(cusparseSetStream(handle,stream));
        cusparseSpMatDescr_t mat; cusparseDnVecDescr_t vx,vy;
        sparse(cusparseCreateCsr(&mat,host.nrows,host.ncols,host.nnz,device.row_ptr,device.col_idx,device.val,
                                CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
        sparse(cusparseCreateDnVec(&vx,host.ncols,dx,CUDA_R_64F)); sparse(cusparseCreateDnVec(&vy,host.nrows,dy,CUDA_R_64F));
        const double alpha=1,beta=0; std::size_t bytes;
        sparse(cusparseSpMV_bufferSize(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,vx,&beta,vy,
                                      CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,&bytes));
        void* buffer; check(cudaMalloc(&buffer,std::max(std::size_t(1),bytes)));
        auto t=std::chrono::steady_clock::now();
        sparse(cusparseSpMV_preprocess(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,vx,&beta,vy,
                                      CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,buffer)); check(cudaStreamSynchronize(stream));
        Candidate c; c.kind="cusparse_alg1"; c.setup_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count();
        c.launch=[&] { sparse(cusparseSpMV(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,vx,&beta,vy,
                                         CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,buffer)); }; candidates.push_back(c);
        std::vector<int> order(candidates.size()); std::iota(order.begin(),order.end(),0);
        std::mt19937 random(seed); std::shuffle(order.begin(),order.end(),random);
        for (int id:order) {
            auto& candidate=candidates[id];
            check(cudaMemsetAsync(dy,0xff,output.size()*sizeof(double),stream)); candidate.launch(); candidate.launch();
            check(cudaStreamSynchronize(stream)); check(cudaMemcpy(output.data(),dy,output.size()*sizeof(double),cudaMemcpyDeviceToHost));
            for (int row=0;row<host.nrows;++row) {
                const long double bound=8*std::numeric_limits<double>::epsilon()*(host.row_ptr[row+1]-host.row_ptr[row]+32.L)*magnitude[row]+1.e-12L;
                if (!std::isfinite(output[row]) || std::abs(output[row]-reference[row])>bound)
                    throw std::runtime_error("incorrect "+candidate.kind+" row="+std::to_string(row));
            }
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            for(int j=0;j<nodes;++j) candidate.launch();
            check(cudaStreamEndCapture(stream,&candidate.graph));
            check(cudaGraphInstantiate(&candidate.executable,candidate.graph,nullptr,nullptr,0));
        }
        cudaEvent_t start,stop; check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
        // Interleave candidates each round to reduce thermal/clock/order bias.
        for(int batch=-2;batch<batches;++batch) {
            std::shuffle(order.begin(),order.end(),random);
            for(int id:order) {
                auto& candidate=candidates[id];
                check(cudaGraphLaunch(candidate.executable,stream)); // cache warm-up
                check(cudaEventRecord(start,stream)); check(cudaGraphLaunch(candidate.executable,stream));
                check(cudaEventRecord(stop,stream)); check(cudaEventSynchronize(stop));
                float ms; check(cudaEventElapsedTime(&ms,start,stop));
                if(batch>=0) candidate.timings.push_back(ms*1000./nodes);
            }
        }
        std::cout<<std::setprecision(9)<<"matrix,rows,cols,nnz,kind,m,balanced,cluster,blocks,shared_per_block,setup_us,median_us,p10_us,p90_us,correct,seed,nodes,batches,warps\n";
        for(auto& candidate:candidates) {
            std::sort(candidate.timings.begin(),candidate.timings.end());
            std::cout<<std::filesystem::path(argv[1]).stem().string()<<','<<host.nrows<<','<<host.ncols<<','<<host.nnz<<','
                     <<candidate.kind<<','<<(candidate.tile<0?0:1<<candidate.tile)<<','<<candidate.balanced<<','<<candidate.cluster<<','
                     <<candidate.blocks<<','<<candidate.shared<<','<<candidate.setup_us<<','<<candidate.timings[5]<<','
                     <<candidate.timings[1]<<','<<candidate.timings[9]<<",1,"<<seed<<','<<nodes<<','<<batches<<','<<candidate.warps<<'\n';
            check(cudaGraphExecDestroy(candidate.executable));check(cudaGraphDestroy(candidate.graph));
        }
        check(cudaEventDestroy(start));check(cudaEventDestroy(stop));
        sparse(cusparseDestroySpMat(mat));sparse(cusparseDestroyDnVec(vx));sparse(cusparseDestroyDnVec(vy));sparse(cusparseDestroy(handle));
        check(cudaFree(buffer));check(cudaStreamDestroy(stream)); for(auto& a:packed) release(a);
        for(void* p:{static_cast<void*>(device.row_ptr),static_cast<void*>(device.col_idx),static_cast<void*>(device.val),static_cast<void*>(dx),static_cast<void*>(dy)}) check(cudaFree(p));
        delete[] host.row_ptr;delete[] host.col_idx;delete[] host.val;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
