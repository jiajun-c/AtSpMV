#include "dsmLongRow.cuh"
#include "csrReader.h"
#include "atspmvprod.h"
#include <set>
#include <cusparse.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>
static void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void sparse(cusparseStatus_t e) { if(e!=CUSPARSE_STATUS_SUCCESS) throw std::runtime_error(cusparseGetErrorString(e)); }
template<class T> T* upload(const std::vector<T>& v) { T* p;check(cudaMalloc(&p,std::max(size_t(1),v.size())*sizeof(T))); if(!v.empty())check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); return p; }
struct Candidate {
    std::string name;
    int group = 1, threshold = 0, tile = -1, balanced = -1, long_rows = 0;
    int threads = 0, blocks = 0, resident_limit = 0;
    std::function<void()> launch;
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    double error = 0;
    std::vector<double> times;
};

int main(int argc, char** argv) { try {
    if(argc<3 || argc>5) throw std::invalid_argument("Usage: dsm_long_row_bench [mixed:|random:]ROWS NNZ_PER_ROW [seed] [--check-only|--sanitizer-only] OR MATRIX.mtx 0 [seed] [--check-only]");
    const unsigned seed=argc>3?std::stoul(argv[3]):202610301;
    const bool sanitizer_only=argc==5 && std::string(argv[4])=="--sanitizer-only";
    const bool check_only=sanitizer_only || (argc==5 && std::string(argv[4])=="--check-only");
    if(argc==5 && !check_only)throw std::invalid_argument("unknown option");
    int device; check(cudaGetDevice(&device)); cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop,device));
    if(prop.major<9) {std::cout<<"SKIP: cluster-capable GPU required\n";return 77;}
    for (const char* key : {"ATSPMV_TUNE_CAP", "ATSPMV_TUNE_WARPS", "ATSPMV_TUNE_PREFETCH", "ATSPMV_TUNE_CTA_REDUCE"}) unsetenv(key);
    int m,n; std::vector<int> ptr,col; std::vector<double> a;
    if(std::string(argv[1]).find(".mtx")!=std::string::npos) {
        auto h=read_mtx(argv[1]); m=h.nrows;n=h.ncols;
        ptr.assign(h.row_ptr,h.row_ptr+m+1);col.assign(h.col_idx,h.col_idx+h.nnz);a.assign(h.val,h.val+h.nnz);
        delete[]h.row_ptr;delete[]h.col_idx;delete[]h.val;
    }else{
        const std::string spec=argv[1];
        const bool mixed=spec.rfind("mixed:",0)==0;
        const bool random_columns=spec.rfind("random:",0)==0;
        m=std::stoi(mixed?spec.substr(6):random_columns?spec.substr(7):spec);int length=std::stoi(argv[2]);
        if(m<=0||length<0||(long long)m*length>40000000) throw std::invalid_argument("invalid synthetic dimensions");
        n=std::max(1024,length*4);ptr.push_back(0);
        for(int r=0;r<m;++r) {
            const int count=mixed ? (r%5==0 ? length-r%17 : r%97) : length;
            for(int j=0;j<std::max(0,count);++j) { col.push_back((j*(random_columns?104729LL:4LL)+r*17)%n);a.push_back(((j*17LL+r*7)%127-63)/79.); }
            ptr.push_back(col.size());
        }
    }
    if(m<=0||n<=0)throw std::invalid_argument("positive matrix dimensions required");
    std::vector<double> x(n),output(m);std::vector<long double> reference(m),magnitude(m);
    for(int j=0;j<n;++j)x[j]=((j*23LL)%97-48)/61.;
    for(int r=0;r<m;++r)for(int j=ptr[r];j<ptr[r+1];++j) {
        long double v=(long double)a[j]*x[col[j]];reference[r]+=v;magnitude[r]+=std::abs(v);
    }
    int* dp=upload(ptr);int* dc=upload(col);double* da=upload(a);double* dx=upload(x);double* dy=upload(output);
    double* scratch; check(cudaMalloc(&scratch,size_t(m)*32*sizeof(double)));
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    std::vector<Candidate> candidates;
    std::vector<int*> lists;
    std::set<int> seen_long_counts;
    for(int threshold:{0,1024,4096}) {
    if(threshold==0 && m>16384) continue;
    std::vector<int> longs,shorts;
    for(int r=0;r<m;++r)(ptr[r+1]-ptr[r]>=threshold?longs:shorts).push_back(r);
    if(!seen_long_counts.insert(longs.size()).second)continue;
    // Sort long rows by length to reduce imbalance inside multi-row clusters.
    std::stable_sort(longs.begin(),longs.end(),[&](int l,int r){return ptr[l+1]-ptr[l]>ptr[r+1]-ptr[r];});
    int* dl=upload(longs);int* ds=upload(shorts);lists.push_back(dl);lists.push_back(ds);
    const int long_count=longs.size(),short_count=shorts.size();
    for(int group:{1,4,8})for(int threads:{128,256,512,1024})for(int parts:{1,2,4,8,16,32})for(int mode:{0,1,2,3,4}) {
        if(threads/group<32)continue;
        if(sanitizer_only && (mode<3 || (threads!=128 && threads!=1024) || (parts!=2 && parts!=8)))continue;
        if((mode==0 && parts!=1)||(mode!=0 && parts==1)||(mode>=3 && parts>8))continue;
        Candidate c;c.long_rows=long_count;c.threshold=threshold;c.name=mode==0?"one_cta":mode==1?"atomic":mode==2?"two_stage":mode==3?"dsm_pull":"dsm_async";
        c.group=group;c.threads=threads;c.blocks=((long_count+group-1)/group)*parts+(mode>=3 ? ((short_count+threads/32*parts-1)/(threads/32*parts))*parts : (short_count+threads/32-1)/(threads/32));c.resident_limit=parts;
        auto bind = [&](auto tag) {
            constexpr int G = decltype(tag)::value;
            c.launch=[=] {
                using namespace atspmv::long_row_detail;
                const int blocks=((long_count+G-1)/G)*parts+(short_count+threads/32-1)/(threads/32);
                if(mode==0)reduceRows<0,G><<<blocks,threads,0,stream>>>(dp,dc,da,dx,dy,scratch,1,dl,long_count,ds,short_count);
                if(mode==1){check(cudaMemsetAsync(dy,0,m*sizeof(double),stream));reduceRows<1,G><<<blocks,threads,0,stream>>>(dp,dc,da,dx,dy,scratch,parts,dl,long_count,ds,short_count);}
                if(mode==2){reduceRows<2,G><<<blocks,threads,0,stream>>>(dp,dc,da,dx,dy,scratch,parts,dl,long_count,ds,short_count);if(long_count)finish<<<(long_count+7)/8,256,0,stream>>>(scratch,dy,long_count,parts,dl);}
                if(mode==3)atspmv::launchDsmLongRows<G,false>(dp,dc,da,dx,dy,long_count,parts,threads,stream,dl,ds,short_count);
                if(mode==4)atspmv::launchDsmLongRows<G,true>(dp,dc,da,dx,dy,long_count,parts,threads,stream,dl,ds,short_count);
                check(cudaGetLastError());
            };
        };
        if(group==1)bind(std::integral_constant<int,1>{});
        if(group==4)bind(std::integral_constant<int,4>{});
        if(group==8)bind(std::integral_constant<int,8>{});
        candidates.push_back(c);
    }
    }
    CSRMatrix<double,int32_t> csr{};
    csr.nrows=m;csr.ncols=n;csr.nnz=a.size();csr.row_ptr=dp;csr.col_idx=dc;csr.val=da;
    std::vector<AtBsr<double,int32_t>> packed(12);
    if(!sanitizer_only && !a.empty())for(int id=0;id<12;++id) {
        CSRToAtBSRProduct(csr,packed[id],id/2,id%2);
        Candidate c;c.name="atspmv";c.tile=1<<(id/2);c.balanced=id%2;
        c.launch=[&,id]{AtSpmvProduct(packed[id],dx,dy,1,0,stream);};candidates.push_back(c);
    }
    cusparseHandle_t handle; sparse(cusparseCreate(&handle));sparse(cusparseSetStream(handle,stream));
    cusparseSpMatDescr_t mat;cusparseDnVecDescr_t vx,vy;
    sparse(cusparseCreateCsr(&mat,m,n,a.size(),dp,dc,da,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
    sparse(cusparseCreateDnVec(&vx,n,dx,CUDA_R_64F));sparse(cusparseCreateDnVec(&vy,m,dy,CUDA_R_64F));
    double alpha=1,beta=0;size_t bytes;void* buffer;
    sparse(cusparseSpMV_bufferSize(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,vx,&beta,vy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,&bytes));
    check(cudaMalloc(&buffer,std::max(size_t(1),bytes)));
    sparse(cusparseSpMV_preprocess(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,vx,&beta,vy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,buffer));
    Candidate cs;cs.name="cusparse";cs.launch=[&]{sparse(cusparseSpMV(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,vx,&beta,vy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,buffer));};candidates.push_back(cs);
        constexpr int nodes = 8, batches = 15;
        auto verify = [&](Candidate& c) {
                check(cudaStreamSynchronize(stream));
                check(cudaMemcpy(output.data(), dy, output.size() * sizeof(double), cudaMemcpyDeviceToHost));
                for (int row = 0; row < m; ++row) {
                    const long double error = std::abs(output[row] - reference[row]);
                    const long double bound = 8 * std::numeric_limits<double>::epsilon() * ((ptr[row+1]-ptr[row]) + 32.L) * magnitude[row] + 1e-12L;
                    if (!std::isfinite(output[row]) || error > bound) throw std::runtime_error("incorrect " + c.name);
                    c.error = std::max(c.error, (double)error);
                }
            };
        for (auto& c : candidates) {
            if(sanitizer_only)std::cerr<<"CHECK "<<c.name<<" G="<<c.group<<" T="<<c.threads<<" C="<<c.resident_limit<<" cutoff="<<c.threshold<<std::endl;
            check(cudaMemsetAsync(dy, 0xff, output.size() * sizeof(double), stream));
            c.launch(); verify(c);
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            for (int k = 0; k < nodes; ++k) c.launch();
            check(cudaStreamEndCapture(stream, &c.graph));
            check(cudaGraphInstantiate(&c.executable, c.graph, nullptr, nullptr, 0));
            check(cudaMemsetAsync(dy, 0xff, output.size() * sizeof(double), stream));
            check(cudaGraphLaunch(c.executable, stream)); verify(c);
        }
        cudaEvent_t start, stop;
        check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
        std::vector<int> order(candidates.size());
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 random(seed);
        for (int batch = -3; !check_only && batch < batches; ++batch) {
            std::shuffle(order.begin(), order.end(), random);
            for (int id : order) {
                auto& c = candidates[id];
                check(cudaGraphLaunch(c.executable, stream));
                check(cudaEventRecord(start, stream));
                check(cudaGraphLaunch(c.executable, stream));
                check(cudaEventRecord(stop, stream)); check(cudaEventSynchronize(stop));
                float ms; check(cudaEventElapsedTime(&ms, start, stop));
                if (batch >= 0) c.times.push_back(ms * 1000. / nodes);
            }
        }
        // A new nonuniform x must be observed by both direct launches and
        // already instantiated graphs; beta=0 must overwrite poisoned output.
        for(int j=0;j<n;++j)x[j]=((j*31LL+7)%113-56)/67.;
        std::fill(reference.begin(),reference.end(),0);
        std::fill(magnitude.begin(),magnitude.end(),0);
        for(int r=0;r<m;++r)for(int j=ptr[r];j<ptr[r+1];++j) {
            long double v=(long double)a[j]*x[col[j]];reference[r]+=v;magnitude[r]+=std::abs(v);
        }
        check(cudaMemcpyAsync(dx,x.data(),n*sizeof(double),cudaMemcpyHostToDevice,stream));
        for(auto& c:candidates) {
            check(cudaMemsetAsync(dy,0xff,m*sizeof(double),stream));c.launch();verify(c);
            check(cudaMemsetAsync(dy,0xff,m*sizeof(double),stream));check(cudaGraphLaunch(c.executable,stream));verify(c);
        }
        if (!check_only) std::cout << "m,n,nnz,sm_count,kind,threads,blocks,parts,long_rows,median_us,p10_us,p90_us,max_abs_error,seed,rows_per_group,threshold,tile,balanced\n";
        for (auto& c : candidates) {
            if (!check_only) {
                std::sort(c.times.begin(), c.times.end());
                std::cout << std::setprecision(10) << m << ',' << n << ',' << a.size() << ',' << prop.multiProcessorCount << ','
                          << c.name << ',' << c.threads << ',' << c.blocks << ',' << c.resident_limit << ',' << c.long_rows << ','
                          << c.times[7] << ',' << c.times[1] << ',' << c.times[13] << ',' << c.error << ',' << seed << ',' << c.group << ',' << c.threshold << ',' << c.tile << ',' << c.balanced << '\n';
            }
            check(cudaGraphExecDestroy(c.executable)); check(cudaGraphDestroy(c.graph));
        }
        if (check_only) std::cout << "PASS: " << candidates.size() << " CSR long-row candidates, two X vectors, direct + graph each\n";
        check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
        sparse(cusparseDestroyDnVec(vx)); sparse(cusparseDestroyDnVec(vy)); sparse(cusparseDestroySpMat(mat)); sparse(cusparseDestroy(handle)); check(cudaFree(buffer)); check(cudaFree(dp)); check(cudaFree(dc)); check(cudaFree(scratch)); check(cudaStreamDestroy(stream));
        for(auto& p:packed)for(void* q:{static_cast<void*>(p.blcVal),static_cast<void*>(p.blcIdx),static_cast<void*>(p.blcIdxInt16),static_cast<void*>(p.blcPtr),static_cast<void*>(p.recorder),static_cast<void*>(p.targetRowByWarp),static_cast<void*>(p.warpBlcPtr)})check(cudaFree(q));
        for(auto p:lists)check(cudaFree(p));
        check(cudaFree(da)); check(cudaFree(dx)); check(cudaFree(dy));
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
