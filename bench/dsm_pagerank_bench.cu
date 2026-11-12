#include "dsmPageRank.h"
#include "csrReader.h"
#include <cusparse.h>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <numeric>
#include <random>
#include <limits>
#include <cmath>

static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void sparse(cusparseStatus_t e){if(e!=CUSPARSE_STATUS_SUCCESS)throw std::runtime_error(cusparseGetErrorString(e));}
template<class T>T* upload(const std::vector<T>& v){T* p;check(cudaMalloc(&p,std::max(std::size_t(1),v.size())*sizeof(T)));if(!v.empty())check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
__global__ void danglingMass(const double* x,const int* mask,int n,double* mass){
    __shared__ double tmp[256];double sum=0;
    for(int i=threadIdx.x;i<n;i+=blockDim.x)if(mask[i])sum+=x[i];
    tmp[threadIdx.x]=sum;__syncthreads();
    for(int s=128;s;s/=2){if(threadIdx.x<s)tmp[threadIdx.x]+=tmp[threadIdx.x+s];__syncthreads();}
    if(!threadIdx.x)*mass=tmp[0];
}
template<int K> __global__ void csrStep(const int* ptr,const int* cols,const double* values,const double* x,double* y,
                        int n,double damping,double inverse_n,const double* mass,bool has_dangling){
    int row=(blockIdx.x*blockDim.x+threadIdx.x)/K,lane=threadIdx.x%K;
    double sum=0;
    if(row<n)for(int j=ptr[row]+lane;j<ptr[row+1];j+=K)sum+=values[j]*x[cols[j]];
    for(int off=K/2;off;off/=2)sum+=__shfl_down_sync(0xffffffff,sum,off,K);
    if(row<n&&lane==0)y[row]=damping*sum+(1-damping+(has_dangling?damping*(*mass):0))*inverse_n;
}
__global__ void updateRank(double* x,int n,double damping,double inverse_n,const double* mass,bool has_dangling){
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<n)x[row]=damping*x[row]+(1-damping+(has_dangling?damping*(*mass):0))*inverse_n;
}
struct Candidate{std::string kind;int cluster=0;std::size_t shared=0;double setup=0,error=0;std::function<void()> launch;cudaGraph_t graph{};cudaGraphExec_t executable{};std::vector<double> times;};
int main(int argc,char** argv){try{
    if(argc<2||argc>5)throw std::invalid_argument("Usage: dsm_pagerank_bench MATRIX.mtx|--synthetic[:N] [iterations=100] [seed=20261030] [--check-only]");
    int iterations=argc>2?std::stoi(argv[2]):100;unsigned seed=argc>3?std::stoul(argv[3]):20261030;
    if(iterations<0||iterations>10000)throw std::invalid_argument("invalid iterations");
    const bool check_only=argc==5&&std::string(argv[4])=="--check-only";
    if(argc==5&&!check_only)throw std::invalid_argument("unknown check-only option");
    int device,supported;check(cudaGetDevice(&device));check(cudaDeviceGetAttribute(&supported,cudaDevAttrClusterLaunch,device));
    if(!supported){std::cout<<"SKIP: cluster-capable GPU required\n";return 77;}
    const double damping=.85;
    std::string name=std::filesystem::path(argv[1]).stem().string();
    int n=0;std::vector<int> rows,columns;
    const std::string input=argv[1];
    const bool synthetic=input=="--synthetic"||input.rfind("--synthetic:",0)==0;
    if(synthetic){
        n=input=="--synthetic"?137:std::stoi(input.substr(12));
        if(n<1||n>100000)throw std::invalid_argument("synthetic size must be in [1,100000]");
        rows.push_back(0);
        // A hub creates a long incoming CSR row; multiples of nine dangle.
        for(int row=0;row<n;++row){if(row%9)for(int j=0;j<1+row%17;++j)columns.push_back(j==0?0:(row*19+j*71)%n);rows.push_back(columns.size());}
    }else{
        auto h=read_mtx(argv[1]);if(h.nrows!=h.ncols||h.nrows<=0)throw std::invalid_argument("PageRank needs a nonempty square graph");
        n=h.nrows;rows.assign(h.row_ptr,h.row_ptr+n+1);columns.assign(h.col_idx,h.col_idx+h.nnz);
        delete[]h.row_ptr;delete[]h.col_idx;delete[]h.val;
    }
    // Unweighted directed PageRank on the input sparsity graph, as in the
    // existing application. Correctly redistribute mass from dangling nodes.
    std::vector<int> ptr(n+1),col(columns.size()),mask(n);
    std::vector<double> val(columns.size());bool has_dangling=false;
    for(int dst:columns)++ptr[dst+1];for(int i=0;i<n;++i)ptr[i+1]+=ptr[i];
    auto cursor=ptr;
    for(int src=0;src<n;++src){int degree=rows[src+1]-rows[src];mask[src]=degree==0;has_dangling|=mask[src];
        for(int j=rows[src];j<rows[src+1];++j){int at=cursor[columns[j]]++;col[at]=src;val[at]=1./degree;}}
    std::vector<double> initial(n,1./n),output(n);
    if(synthetic){
        double total=0;
        for(int i=0;i<n;++i){initial[i]=1+(i*13)%31;total+=initial[i];}
        for(double& x:initial)x/=total;
    }
    std::vector<long double> reference(initial.begin(),initial.end()),next(n);
    for(int it=0;it<iterations;++it){long double mass=0;for(int i=0;i<n;++i)if(mask[i])mass+=reference[i];
        for(int row=0;row<n;++row){long double sum=0;for(int j=ptr[row];j<ptr[row+1];++j)sum+=val[j]*reference[col[j]];
            next[row]=damping*sum+(1-damping+damping*mass)/n;}reference.swap(next);}
    CSRMatrix<double,int32_t>a;a.nrows=a.ncols=n;a.nnz=col.size();a.row_ptr=upload(ptr);a.col_idx=upload(col);a.val=upload(val);
    int* dmask=upload(mask);double* dx=upload(initial);double* dy=upload(output);double* work[2]={upload(output),upload(output)};double* mass=upload(std::vector<double>(1));
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    std::vector<Candidate> candidates;
    for(int k:{1,4,32}) {
    Candidate fused;fused.kind="fused_csr_graph_k"+std::to_string(k);
    fused.launch=[&,k]{check(cudaMemcpyAsync(work[0],dx,n*sizeof(double),cudaMemcpyDeviceToDevice,stream));
        for(int i=0;i<iterations;++i){if(has_dangling)danglingMass<<<1,256,0,stream>>>(work[i%2],dmask,n,mass);
            if(k==1)csrStep<1><<<(n+255)/256,256,0,stream>>>(a.row_ptr,a.col_idx,a.val,work[i%2],work[1-i%2],n,damping,1.0/n,mass,has_dangling);
            else if(k==4)csrStep<4><<<(n+63)/64,256,0,stream>>>(a.row_ptr,a.col_idx,a.val,work[i%2],work[1-i%2],n,damping,1.0/n,mass,has_dangling);
            else csrStep<32><<<(n+7)/8,256,0,stream>>>(a.row_ptr,a.col_idx,a.val,work[i%2],work[1-i%2],n,damping,1.0/n,mass,has_dangling);}
        check(cudaMemcpyAsync(dy,work[iterations%2],n*sizeof(double),cudaMemcpyDeviceToDevice,stream));};candidates.push_back(fused);
    }
    cusparseHandle_t handle;sparse(cusparseCreate(&handle));sparse(cusparseSetStream(handle,stream));
    cusparseSpMatDescr_t mat;cusparseDnVecDescr_t vx,vy;
    sparse(cusparseCreateCsr(&mat,n,n,a.nnz,a.row_ptr,a.col_idx,a.val,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
    sparse(cusparseCreateDnVec(&vx,n,work[0],CUDA_R_64F));sparse(cusparseCreateDnVec(&vy,n,work[1],CUDA_R_64F));
    const double alpha=1,beta=0;std::size_t bytes;
    sparse(cusparseSpMV_bufferSize(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,vx,&beta,vy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,&bytes));
    void* buffer;check(cudaMalloc(&buffer,std::max(std::size_t(1),bytes)));
    sparse(cusparseSpMV_preprocess(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,vx,&beta,vy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,buffer));check(cudaStreamSynchronize(stream));
    Candidate cu;cu.kind="cusparse_graph";cu.launch=[&]{check(cudaMemcpyAsync(work[0],dx,n*sizeof(double),cudaMemcpyDeviceToDevice,stream));
        for(int i=0;i<iterations;++i){if(has_dangling)danglingMass<<<1,256,0,stream>>>(work[i%2],dmask,n,mass);
            sparse(cusparseDnVecSetValues(vx,work[i%2]));sparse(cusparseDnVecSetValues(vy,work[1-i%2]));
            sparse(cusparseSpMV(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,vx,&beta,vy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,buffer));
            updateRank<<<(n+255)/256,256,0,stream>>>(work[1-i%2],n,damping,1.0/n,mass,has_dangling);}
        check(cudaMemcpyAsync(dy,work[iterations%2],n*sizeof(double),cudaMemcpyDeviceToDevice,stream));};candidates.push_back(cu);
    for(int cluster:{1,2,4,8})for(auto storage:{atspmv::XStorage::Global,atspmv::XStorage::DistributedShared}){
        auto t=std::chrono::steady_clock::now();
        try{auto plan=atspmv::makePersistentPageRankPlan(a,dmask,has_dangling,cluster,storage);
            Candidate c;c.kind=storage==atspmv::XStorage::Global?"persistent_global":"persistent_dsm";c.cluster=cluster;c.shared=plan.sharedBytesPerBlock();
            c.setup=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count();
            c.launch=[&,plan]{atspmv::launchPersistentPageRank(plan,dx,dy,iterations,damping,stream);};candidates.push_back(c);
        }catch(const std::invalid_argument& e){if(std::string(e.what()).find("exceed cluster shared capacity")==std::string::npos)throw;}}
    for(auto& c:candidates){check(cudaMemsetAsync(dy,0xff,n*sizeof(double),stream));c.launch();check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(output.data(),dy,n*sizeof(double),cudaMemcpyDeviceToHost));double total=0;
        for(int i=0;i<n;++i){c.error=std::max(c.error,double(std::abs(output[i]-reference[i])));total+=output[i];
            if(!std::isfinite(output[i])||output[i]<0||std::abs(output[i]-reference[i])>1e-11)throw std::runtime_error("incorrect "+c.kind);}
        if(std::abs(total-1)>1e-10)throw std::runtime_error("PageRank mass not conserved");
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));c.launch();check(cudaStreamEndCapture(stream,&c.graph));
        check(cudaGraphInstantiate(&c.executable,c.graph,nullptr,nullptr,0));
        check(cudaGraphLaunch(c.executable,stream));check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(output.data(),dy,n*sizeof(double),cudaMemcpyDeviceToHost));
        for(int i=0;i<n;++i)if(!std::isfinite(output[i])||std::abs(output[i]-reference[i])>1e-11)
            throw std::runtime_error("incorrect graph "+c.kind);
    }
    cudaEvent_t start,stop;check(cudaEventCreate(&start));check(cudaEventCreate(&stop));
    std::vector<int> order(candidates.size());std::iota(order.begin(),order.end(),0);std::mt19937 random(seed);
    for(int batch=-3;!check_only&&batch<15;++batch){std::shuffle(order.begin(),order.end(),random);for(int id:order){auto& c=candidates[id];
        check(cudaGraphLaunch(c.executable,stream));check(cudaEventRecord(start,stream));check(cudaGraphLaunch(c.executable,stream));
        check(cudaEventRecord(stop,stream));check(cudaEventSynchronize(stop));float ms;check(cudaEventElapsedTime(&ms,start,stop));if(batch>=0)c.times.push_back(ms*1000.);}}
    if(!check_only)std::cout<<std::setprecision(10)<<"matrix,n,nnz,iterations,dangling,kind,cluster,shared_per_block,setup_us,median_us,p10_us,p90_us,max_abs_error,seed\n";
    for(auto& c:candidates){if(!check_only){std::sort(c.times.begin(),c.times.end());std::cout<<name<<','<<n<<','<<a.nnz<<','<<iterations<<','<<has_dangling<<','<<c.kind<<','<<c.cluster<<','<<c.shared<<','<<c.setup<<','<<c.times[7]<<','<<c.times[1]<<','<<c.times[13]<<','<<c.error<<','<<seed<<'\n';}
        check(cudaGraphExecDestroy(c.executable));check(cudaGraphDestroy(c.graph));}
    if(check_only)std::cout<<"PASS: "<<candidates.size()<<" direct and graph candidates, iterations="<<iterations<<"\n";
    candidates.clear(); // release plan-owned workspace while its device is current
    sparse(cusparseDestroySpMat(mat));sparse(cusparseDestroyDnVec(vx));sparse(cusparseDestroyDnVec(vy));sparse(cusparseDestroy(handle));check(cudaFree(buffer));
    check(cudaEventDestroy(start));check(cudaEventDestroy(stop));check(cudaStreamDestroy(stream));
    for(void* p:{static_cast<void*>(a.row_ptr),static_cast<void*>(a.col_idx),static_cast<void*>(a.val),static_cast<void*>(dmask),static_cast<void*>(dx),static_cast<void*>(dy),static_cast<void*>(work[0]),static_cast<void*>(work[1]),static_cast<void*>(mass)})check(cudaFree(p));
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
