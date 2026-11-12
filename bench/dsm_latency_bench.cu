#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
namespace cg=cooperative_groups;
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__device__ __forceinline__ unsigned smid(){unsigned v;asm volatile("mov.u32 %0, %%smid;":"=r"(v));return v;}
__device__ __forceinline__ unsigned long long ticks(){unsigned long long v;asm volatile("mov.u64 %0, %%clock64;":"=l"(v)::"memory");return v;}
// One active lane: each returned index determines the next address. The global
// load uses .cg to bypass L1; repeated small working set is intended to hit L2.
// The two shared variants use identical index data and pointer arithmetic.
template<int Kind>
__global__ void chase(const unsigned* global,unsigned long long* clocks,unsigned* sink,unsigned* sms,int iterations) {
    __shared__ unsigned table[1024];
    for(int i=threadIdx.x;i<1024;i+=blockDim.x)table[i]=(i+73)&1023;
    auto cluster=cg::this_cluster();
    if(threadIdx.x==0)sms[cluster.block_rank()]=smid();
    cluster.sync();
    if(cluster.block_rank()==0&&threadIdx.x==0){
        unsigned idx=0;
        unsigned base=static_cast<unsigned>(__cvta_generic_to_shared(table));
        if constexpr(Kind==2)asm("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(base):"r"(base),"r"(1));
        unsigned long long begin=ticks();
        for(int i=0;i<iterations;++i){
            if constexpr(Kind==0) idx=(idx+73)&1023;
            if constexpr(Kind==1) {unsigned addr=base+idx*4;asm volatile("ld.shared.u32 %0, [%1];":"=r"(idx):"r"(addr):"memory");}
            if constexpr(Kind==2) {unsigned addr=base+idx*4;asm volatile("ld.shared::cluster.u32 %0, [%1];":"=r"(idx):"r"(addr):"memory");}
            if constexpr(Kind==3) {const unsigned* addr=global+idx;asm volatile("ld.global.cg.u32 %0, [%1];":"=r"(idx):"l"(addr):"memory");}
        }
        unsigned long long end=ticks();*clocks=end-begin;*sink=idx;
    }
    cluster.sync(); // Remote table must stay alive through the read chain.
}
// Each atomic's returned old value determines the next addend. Thus it cannot
// be compiled as returnless RED, nor can independent atomics overlap per lane.
// shared_address=true makes all CTAs update the same global FP64 address.
template<bool ReturnOld, bool SyncStart>
__global__ void atomicChain(double* values,unsigned long long* clocks,unsigned* sms,int iterations,bool shared_address) {
    const int rank=blockIdx.x;
    if(threadIdx.x==0)sms[rank]=smid();
    if constexpr(SyncStart) cg::this_cluster().sync();
    if(threadIdx.x==0){
        double previous=0;double* p=values+(shared_address?0:rank*16);
        unsigned long long begin=ticks();
        for(int i=0;i<iterations;++i){
            if constexpr(ReturnOld) {
                const double increment=previous>=0.0?1.0:-1.0;
                previous=atomicAdd(p,increment);
            } else {
                atomicAdd(p,1.0); // Production-style unused return; may lower to RED.
            }
        }
        unsigned long long end=ticks();clocks[rank]=end-begin;
    }
}
// Control: two back-to-back clock reads, including only the surrounding
// single-thread branch and result store outside the timed interval.
template<bool SyncStart>
__global__ void clockPair(unsigned long long* clocks,unsigned* sms){
    const int rank=blockIdx.x;
    if(threadIdx.x==0)sms[rank]=smid();
    if constexpr(SyncStart)cg::this_cluster().sync();
    if(threadIdx.x==0){unsigned long long a=ticks();unsigned long long b=ticks();clocks[rank]=b-a;}
}
// FP64 CUDA atomicAdd into local shared or CTA1's mapped DSM. This exercises
// identical atomic semantics, but shared FP64 may be implemented with CAS;
// inspect the generated SASS before comparing with a native global ATOM.
__global__ void sharedAtomicChain(unsigned long long* clocks,double* output,unsigned* sms,int iterations,bool remote){
    __shared__ double value;
    auto cluster=cg::this_cluster();int rank=cluster.block_rank();
    if(threadIdx.x==0){value=0;sms[rank]=smid();}
    cluster.sync();
    if(rank==0&&threadIdx.x==0){
        double* target=remote?cluster.map_shared_rank(&value,1):&value;
        double previous=0;unsigned long long begin=ticks();
        for(int i=0;i<iterations;++i)previous=atomicAdd(target,previous>=0?1.0:-1.0);
        clocks[0]=ticks()-begin;
    }
    cluster.sync();
    if(threadIdx.x==0&&rank==(remote?1:0))*output=value;
}
// One communication round of the production all-to-all protocol. Each warp
// sends one FP64 partial to each CTA; each CTA waits for its own arrivals.
// The measured window begins after cluster initialization and ends on each CTA
// after its mbarrier wait; CTA0 additionally combines and writes the result.
__global__ void asyncRound(unsigned long long* clocks,double* output,unsigned* sms,int parts){
    __shared__ unsigned long long barrier;
    __shared__ double inbox[32*8];
    auto cluster=cg::this_cluster();int rank=cluster.block_rank();
    if(threadIdx.x==0)sms[rank]=smid();
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    unsigned b=static_cast<unsigned>(__cvta_generic_to_shared(&barrier));
    if(threadIdx.x==0)asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(b):"memory");
    asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
    cluster.sync();
    unsigned long long begin=0;
    if(threadIdx.x==0)begin=ticks();
    if(threadIdx.x==0){unsigned bytes=(blockDim.x/32)*parts*8;
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"::"r"(b),"r"(bytes):"memory");}
    if(lane<parts){
        unsigned dest=static_cast<unsigned>(__cvta_generic_to_shared(&inbox[warp*parts+rank])),remote,remote_b;
        asm("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(remote):"r"(dest),"r"(lane));
        asm("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(remote_b):"r"(b),"r"(lane));
        long long bits=__double_as_longlong(1.0);
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b64 [%0], %1, [%2];"::"r"(remote),"l"(bits),"r"(remote_b):"memory");
    }
    unsigned ready;do{asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }":"=r"(ready):"r"(b):"memory");}while(!ready);
    if(rank==0&&warp==0){
        double sum=0;for(int i=lane;i<(blockDim.x/32)*parts;i+=32)sum+=inbox[i];
        for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
        if(lane==0)*output=sum;
    }
    if(threadIdx.x==0)clocks[rank]=ticks()-begin;
}
template<class Kernel,class... Args> void launchCluster(cudaStream_t stream,int parts,int threads,Kernel kernel,Args... args){
    cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={static_cast<unsigned>(parts),1,1};
    cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(parts);cfg.blockDim=dim3(threads);cfg.stream=stream;cfg.attrs=&attr;cfg.numAttrs=1;
    check(cudaLaunchKernelEx(&cfg,kernel,args...));
}
struct Candidate {
    std::string name;int parts=0,threads=0,iterations=0;
    std::function<void()> launch;cudaGraph_t graph{};cudaGraphExec_t exec{};
    std::vector<double> times,cycles,max_cycles;
    int min_sms=99,max_sms=0;
};
int main(int argc,char**argv){try{
    int iterations=argc>1?std::stoi(argv[1]):4096, trials=argc>2?std::stoi(argv[2]):41;
    unsigned seed=argc>3?std::stoul(argv[3]):202610271;
    const bool check_only=argc==5&&std::string(argv[4])=="--check-only";
    if(argc>5||(argc==5&&!check_only)||iterations<2||iterations>100000||trials<5||trials>1001)
        throw std::invalid_argument("Usage: dsm_latency_bench [iterations=4096] [trials=41] [seed] [--check-only]");
    if(check_only)trials=1;
    int device,cluster_ok;cudaDeviceProp prop{};check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&prop,device));
    check(cudaDeviceGetAttribute(&cluster_ok,cudaDevAttrClusterLaunch,device));
    if(!cluster_ok){std::cout<<"SKIP: cluster-capable GPU required\n";return 77;}
    unsigned* dtable;unsigned* sink;unsigned* sms;unsigned long long* clocks;double *values,*output;
    check(cudaMalloc(&dtable,1024*sizeof(unsigned)));check(cudaMalloc(&sink,sizeof(unsigned)));check(cudaMalloc(&sms,8*sizeof(unsigned)));
    check(cudaMalloc(&clocks,8*sizeof(unsigned long long)));check(cudaMalloc(&values,128*sizeof(double)));check(cudaMalloc(&output,sizeof(double)));
    std::vector<unsigned> table(1024);for(int i=0;i<1024;++i)table[i]=(i+73)&1023;
    check(cudaMemcpy(dtable,table.data(),table.size()*sizeof(unsigned),cudaMemcpyHostToDevice));
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    std::vector<Candidate> candidates;
    for(int kind=0;kind<4;++kind){
        Candidate c;c.name=kind==0?"loop_control":kind==1?"local_shared_load":kind==2?"remote_dsm_load":"global_l2_load_cg";
        c.parts=2;c.threads=128;c.iterations=iterations;
        c.launch=[=]{switch(kind){
            case 0:launchCluster(stream,2,128,chase<0>,dtable,clocks,sink,sms,iterations);break;
            case 1:launchCluster(stream,2,128,chase<1>,dtable,clocks,sink,sms,iterations);break;
            case 2:launchCluster(stream,2,128,chase<2>,dtable,clocks,sink,sms,iterations);break;
            case 3:launchCluster(stream,2,128,chase<3>,dtable,clocks,sink,sms,iterations);break;}};
        candidates.push_back(c);
    }
    for(bool remote:{false,true}) {
        Candidate c;c.name=remote?"remote_dsm_fp64_atomic":"local_shared_fp64_atomic";
        c.parts=2;c.threads=128;c.iterations=iterations;
        c.launch=[=]{launchCluster(stream,2,128,sharedAtomicChain,clocks,output,sms,iterations,remote);};
        candidates.push_back(c);
    }
    for(int parts:{1,2,4,8}){
        for(bool cluster_sync:{false,true}) {
            Candidate c;c.name=cluster_sync?"clock_pair_cluster":"clock_pair_plain";
            c.parts=parts;c.threads=128;c.iterations=1;
            c.launch=[=]{
                if(cluster_sync)launchCluster(stream,parts,128,clockPair<true>,clocks,sms);
                else {clockPair<false><<<parts,128,0,stream>>>(clocks,sms);check(cudaGetLastError());}
            };candidates.push_back(c);
        }
        for(bool return_old:{true,false})for(bool shared:{false,true})for(int count:{1,iterations}){
            Candidate c;
            c.name=return_old ? (shared?"atomic_return_contended":"atomic_return_independent")
                              : (shared?"atomic_discard_contended":"atomic_discard_independent");
            c.parts=parts;c.threads=128;c.iterations=count;
            c.launch=[=]{
                if(return_old)launchCluster(stream,parts,128,atomicChain<true,true>,values,clocks,sms,count,shared);
                else launchCluster(stream,parts,128,atomicChain<false,true>,values,clocks,sms,count,shared);
            };candidates.push_back(c);
        }
        // Production baseline: ordinary grid, no cluster launch/barrier.
        for(bool return_old:{true,false})for(bool shared:{false,true})for(int count:{1,iterations}){
            Candidate c;
            c.name=return_old ? (shared?"atomic_plain_return_contended":"atomic_plain_return_independent")
                              : (shared?"atomic_plain_discard_contended":"atomic_plain_discard_independent");
            c.parts=parts;c.threads=128;c.iterations=count;
            c.launch=[=]{
                if(return_old)atomicChain<true,false><<<parts,128,0,stream>>>(values,clocks,sms,count,shared);
                else atomicChain<false,false><<<parts,128,0,stream>>>(values,clocks,sms,count,shared);
                check(cudaGetLastError());
            };candidates.push_back(c);
        }
        if(parts==1)continue;
        Candidate c;c.name="dsm_async_round";c.parts=parts;c.threads=128;c.iterations=1;
        c.launch=[=]{launchCluster(stream,parts,128,asyncRound,clocks,output,sms,parts);};candidates.push_back(c);
    }
    const int nodes=check_only?2:32;
    for(auto& c:candidates){
        check(cudaMemsetAsync(values,0,128*sizeof(double),stream));check(cudaMemsetAsync(clocks,0,8*sizeof(unsigned long long),stream));
        c.launch();check(cudaStreamSynchronize(stream));
        if(c.name.rfind("clock_pair",0)==0){
            unsigned long long v;check(cudaMemcpy(&v,clocks,sizeof(v),cudaMemcpyDeviceToHost));
            if(!v)throw std::runtime_error("invalid clock pair");
        }else if(c.name=="local_shared_fp64_atomic"||c.name=="remote_dsm_fp64_atomic"){
            double v;check(cudaMemcpy(&v,output,sizeof(v),cudaMemcpyDeviceToHost));
            if(v!=c.iterations)throw std::runtime_error("shared FP64 atomic result incorrect");
        }else if(c.name.find("atomic")!=std::string::npos){
            double vals[128];check(cudaMemcpy(vals,values,sizeof(vals),cudaMemcpyDeviceToHost));
            if(c.name.find("contended")!=std::string::npos){if(vals[0]!=double(c.parts)*c.iterations)throw std::runtime_error("atomic result incorrect");}
            else for(int i=0;i<c.parts;++i)if(vals[i*16]!=c.iterations)throw std::runtime_error("atomic independent result incorrect");
        }else if(c.name=="dsm_async_round"){
            double v;check(cudaMemcpy(&v,output,sizeof(v),cudaMemcpyDeviceToHost));
            if(v!=c.parts*(c.threads/32))throw std::runtime_error("DSM round result incorrect");
        }else{
            unsigned index;check(cudaMemcpy(&index,sink,sizeof(index),cudaMemcpyDeviceToHost));
            if(index!=((iterations*73)&1023))throw std::runtime_error("pointer chase result incorrect");
        }
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        for(int i=0;i<nodes;++i)c.launch();
        check(cudaStreamEndCapture(stream,&c.graph));check(cudaGraphInstantiate(&c.exec,c.graph,nullptr,nullptr,0));
    }
    cudaEvent_t start,stop;check(cudaEventCreate(&start));check(cudaEventCreate(&stop));
    std::vector<int> order(candidates.size());std::iota(order.begin(),order.end(),0);std::mt19937 rng(seed);
    for(int batch=check_only?0:-3;batch<trials;++batch){
        std::shuffle(order.begin(),order.end(),rng);
        for(int id:order){auto&c=candidates[id];
            check(cudaMemsetAsync(values,0,128*sizeof(double),stream));
            check(cudaGraphLaunch(c.exec,stream)); // warm
            check(cudaEventRecord(start,stream));check(cudaGraphLaunch(c.exec,stream));check(cudaEventRecord(stop,stream));check(cudaEventSynchronize(stop));
            if(batch>=0){
                float ms;check(cudaEventElapsedTime(&ms,start,stop));c.times.push_back(ms*1000./nodes);
                unsigned long long v[8]{};check(cudaMemcpy(v,clocks,c.parts*sizeof(*v),cudaMemcpyDeviceToHost));
                if(v[0]==0)throw std::runtime_error("missing clock sample");
                if(batch==0){
                    if(c.name.rfind("atomic_",0)==0){
                        double vals[128];check(cudaMemcpy(vals,values,sizeof(vals),cudaMemcpyDeviceToHost));
                        if(c.name.find("contended")!=std::string::npos) {
                            if(vals[0]!=double(2*nodes)*c.parts*c.iterations)throw std::runtime_error("atomic graph result incorrect");
                        } else for(int k=0;k<c.parts;++k)if(vals[k*16]!=double(2*nodes)*c.iterations)throw std::runtime_error("atomic graph result incorrect");
                    } else if(c.name.rfind("clock_pair",0)==0) {
                        // Nonzero clock output was checked just above.
                    } else if(c.name=="dsm_async_round"||c.name=="local_shared_fp64_atomic"||c.name=="remote_dsm_fp64_atomic") {
                        double vout;check(cudaMemcpy(&vout,output,sizeof(vout),cudaMemcpyDeviceToHost));
                        const double expected=c.name=="dsm_async_round"?c.parts*(c.threads/32):c.iterations;
                        if(vout!=expected)throw std::runtime_error("shared graph result incorrect");
                    } else {
                        unsigned index;check(cudaMemcpy(&index,sink,sizeof(index),cudaMemcpyDeviceToHost));
                        if(index!=((c.iterations*73)&1023))throw std::runtime_error("chase graph result incorrect");
                    }
                }
                c.cycles.push_back(double(v[0])/c.iterations);
                const int count=c.name.rfind("atomic_",0)==0||c.name.rfind("clock_pair",0)==0||c.name=="dsm_async_round"?c.parts:1;
                c.max_cycles.push_back(double(*std::max_element(v,v+count))/c.iterations);
                unsigned ids[8];check(cudaMemcpy(ids,sms,c.parts*sizeof(unsigned),cudaMemcpyDeviceToHost));
                std::sort(ids,ids+c.parts);int distinct=std::unique(ids,ids+c.parts)-ids;
                c.min_sms=std::min(c.min_sms,distinct);c.max_sms=std::max(c.max_sms,distinct);
            }
        }
    }
    auto q=[](std::vector<double>&v,double f){std::sort(v.begin(),v.end());return v[size_t(f*(v.size()-1))];};
    std::cout<<"kind,cluster,threads,iterations,nodes,trials,seed,sm_count,median_cycles_per_iter,p10_cycles,p90_cycles,median_max_cta_cycles_per_iter,median_kernel_us,p10_kernel_us,p90_kernel_us,min_distinct_sms,max_distinct_sms\n";
    for(auto&c:candidates){
        std::cout<<std::setprecision(10)<<c.name<<','<<c.parts<<','<<c.threads<<','<<c.iterations<<','<<nodes<<','<<trials<<','<<seed<<','<<prop.multiProcessorCount<<','
                 <<q(c.cycles,.5)<<','<<q(c.cycles,.1)<<','<<q(c.cycles,.9)<<','<<q(c.max_cycles,.5)<<','<<q(c.times,.5)<<','<<q(c.times,.1)<<','<<q(c.times,.9)<<','<<c.min_sms<<','<<c.max_sms<<'\n';
        check(cudaGraphExecDestroy(c.exec));check(cudaGraphDestroy(c.graph));
    }
    if(check_only)std::cerr<<"PASS: "<<candidates.size()<<" direct and graph correctness checks\n";
    check(cudaEventDestroy(start));check(cudaEventDestroy(stop));check(cudaStreamDestroy(stream));
    check(cudaFree(dtable));check(cudaFree(sink));check(cudaFree(sms));check(cudaFree(clocks));check(cudaFree(values));check(cudaFree(output));
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
