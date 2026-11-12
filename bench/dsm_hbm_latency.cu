#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
namespace cg = cooperative_groups;
static void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
__device__ __forceinline__ unsigned long long ticks() { unsigned long long t; asm volatile("mov.u64 %0, %%clock64;":"=l"(t)::"memory"); return t; }
// Kind 0: remote shared; 1: hot global; 2: a fresh segment of a large
// random pointer chain. Each dependent load determines the next address.
template<int Kind> __global__ void measure(const unsigned* hot, const unsigned* cold, const unsigned* starts,
 unsigned* sequence, unsigned segments, unsigned long long* cycles, unsigned* sink, unsigned* segment,
 int iters, int peer) {
    __shared__ unsigned table[1024];
    __shared__ unsigned start, seq;
    for (int i=threadIdx.x;i<1024;i+=blockDim.x) table[i]=(i+73)&1023;
    auto cluster=cg::this_cluster();
    if constexpr (Kind==2) if (cluster.block_rank()==0 && threadIdx.x==0) {
        seq=atomicAdd(sequence,1u)%segments; start=starts[seq];
    }
    cluster.sync();
    if (cluster.block_rank()==0 && threadIdx.x==0) {
        unsigned idx=0, base=static_cast<unsigned>(__cvta_generic_to_shared(table));
        if constexpr (Kind==0) asm("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(base):"r"(base),"r"(peer));
        if constexpr (Kind==2) idx=start;
        unsigned long long begin=ticks();
        for (int i=0;i<iters;++i) {
            if constexpr (Kind==0) { unsigned addr=base+idx*4; asm volatile("ld.shared::cluster.u32 %0, [%1];":"=r"(idx):"r"(addr):"memory"); }
            if constexpr (Kind==1) { const unsigned* p=hot+idx; asm volatile("ld.global.cg.u32 %0, [%1];":"=r"(idx):"l"(p):"memory"); }
            if constexpr (Kind==2) { const unsigned* p=cold+static_cast<size_t>(idx)*32; asm volatile("ld.global.cg.u32 %0, [%1];":"=r"(idx):"l"(p):"memory"); }
        }
        *cycles=ticks()-begin; *sink=idx;
        if constexpr (Kind==2) *segment=seq;
    }
    cluster.sync();
}
template<class Kernel, class... Args> void launch(cudaStream_t stream, int c, Kernel kernel, Args...args) {
    cudaLaunchAttribute attr{}; attr.id=cudaLaunchAttributeClusterDimension; attr.val.clusterDim={static_cast<unsigned>(c),1,1};
    cudaLaunchConfig_t cfg{}; cfg.gridDim=dim3(c); cfg.blockDim=dim3(128); cfg.stream=stream; cfg.attrs=&attr; cfg.numAttrs=1;
    check(cudaLaunchKernelEx(&cfg,kernel,args...));
}
template<class Kernel> bool supports16(Kernel k, cudaStream_t stream) {
    if (cudaFuncSetAttribute(k,cudaFuncAttributeNonPortableClusterSizeAllowed,1)!=cudaSuccess) { cudaGetLastError(); return false; }
    cudaLaunchAttribute a{};a.id=cudaLaunchAttributeClusterDimension;a.val.clusterDim={16,1,1};
    cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(16);cfg.blockDim=dim3(128);cfg.stream=stream;cfg.attrs=&a;cfg.numAttrs=1;
    int active=0; auto e=cudaOccupancyMaxActiveClusters(&active,k,&cfg); if(e!=cudaSuccess){cudaGetLastError();return false;} return active>0;
}
struct Case { int kind, cluster, peer; };
int main(int argc,char** argv) { try {
    int iters=argc>1?std::stoi(argv[1]):4096, trials=argc>2?std::stoi(argv[2]):25;
    unsigned seed=argc>3?std::stoul(argv[3]):202610271u; int mib=argc>4?std::stoi(argv[4]):512;
    bool profile=argc>5 && std::string(argv[5])=="--profile";
    if(iters<1 || trials<1 || mib<1 || mib>2048 || (static_cast<unsigned long long>(mib)*8192)%iters) throw std::runtime_error("invalid arguments or chain length");
    int device=0,cap=0; cudaDeviceProp prop{};check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&prop,device));check(cudaDeviceGetAttribute(&cap,cudaDevAttrClusterLaunch,device));
    if(!cap){std::cerr<<"cluster-capable device required\n";return 77;}
    size_t n=static_cast<size_t>(mib)*8192, segments=n/iters;
    std::vector<unsigned> order(n), nodes(n*32,0), starts(segments), hot(1024);
    std::iota(order.begin(),order.end(),0u); std::mt19937 gen(seed);std::shuffle(order.begin(),order.end(),gen);
    for(size_t i=0;i<n;++i) nodes[static_cast<size_t>(order[i])*32]=order[(i+1)%n];
    for(size_t i=0;i<segments;++i) starts[i]=order[i*iters];
    for(unsigned i=0;i<1024;++i)hot[i]=(i+73)&1023;
    unsigned *dhot,*dcold,*dstarts,*seq,*sink,*seg; unsigned long long* cyc;
    check(cudaMalloc(&dhot,4096));check(cudaMalloc(&dcold,n*128));check(cudaMalloc(&dstarts,segments*4));
    check(cudaMalloc(&seq,4));check(cudaMalloc(&sink,4));check(cudaMalloc(&seg,4));check(cudaMalloc(&cyc,8));
    check(cudaMemcpy(dhot,hot.data(),4096,cudaMemcpyHostToDevice));check(cudaMemcpy(dcold,nodes.data(),n*128,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dstarts,starts.data(),segments*4,cudaMemcpyHostToDevice));check(cudaMemset(seq,0,4));
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    bool s0=supports16(measure<0>,stream),s1=supports16(measure<1>,stream),s2=supports16(measure<2>,stream);
    std::vector<int> sizes={2,4,8};if(s0&&s1&&s2)sizes.push_back(16);
    std::vector<Case> cases;
    if(profile) { int c=argc>6?std::stoi(argv[6]):8; if(std::find(sizes.begin(),sizes.end(),c)==sizes.end()) throw std::runtime_error("unsupported profile cluster"); cases.push_back({2,c,-1}); }
    else for(int c:sizes){cases.push_back({0,c,1});if(c>2)cases.push_back({0,c,c-1});cases.push_back({1,c,-1});cases.push_back({2,c,-1});}
    std::cout<<"kind,cluster,peer,trial,cycles,iterations,segment\n";
    for(int trial=0;trial<trials;++trial) {
        if(!profile)std::shuffle(cases.begin(),cases.end(),gen);
        for(auto v:cases){
            if(v.kind==0) launch(stream,v.cluster,measure<0>,dhot,dcold,dstarts,seq,(unsigned)segments,cyc,sink,seg,iters,v.peer);
            else if(v.kind==1) launch(stream,v.cluster,measure<1>,dhot,dcold,dstarts,seq,(unsigned)segments,cyc,sink,seg,iters,v.peer);
            else launch(stream,v.cluster,measure<2>,dhot,dcold,dstarts,seq,(unsigned)segments,cyc,sink,seg,iters,v.peer);
            check(cudaStreamSynchronize(stream));
            unsigned got,segment=0;unsigned long long clock;check(cudaMemcpy(&clock,cyc,8,cudaMemcpyDeviceToHost));check(cudaMemcpy(&got,sink,4,cudaMemcpyDeviceToHost));
            unsigned expected=(static_cast<unsigned>(iters)*73)&1023;
            if(v.kind==2){check(cudaMemcpy(&segment,seg,4,cudaMemcpyDeviceToHost)); expected=starts[(segment+1)%segments];}
            if(got!=expected)throw std::runtime_error("pointer chain correctness failure");
            std::cout<<(v.kind==0?"dsm":v.kind==1?"hot_l2":"cold_global")<<','<<v.cluster<<','<<v.peer<<','<<trial<<','<<clock<<','<<iters<<','<<segment<<'\n';
        }
    }
    check(cudaStreamDestroy(stream));check(cudaFree(dhot));check(cudaFree(dcold));check(cudaFree(dstarts));check(cudaFree(seq));check(cudaFree(sink));check(cudaFree(seg));check(cudaFree(cyc));
    return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;} }
