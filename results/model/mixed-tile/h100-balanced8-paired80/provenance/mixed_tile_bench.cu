// Experimental mixed AtBSR tiles: optimize allocated slots, then compare
// one heterogeneous kernel against each fixed tile/balance combination.
#include "atspmvprod.h"
#include "csrReader.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> static T* upload(const std::vector<T>& v) {
    T* p=nullptr; ck(cudaMalloc(&p,std::max(size_t(1),v.size())*sizeof(T)));
    if(!v.empty())ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); return p;
}
struct Group { int row_start, tile_start, tile_end, m; };
template<class Index>
__global__ void mixed_spmv(const Group* groups,int count,const int* row_ids,int nrows,
                            const Index* index,const double* value,const double* x,double* y) {
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/32, lane=threadIdx.x%32;
    if(warp>=count)return;
    Group g=groups[warp]; int k=32/g.m; double sum=0;
    for(int tile=g.tile_start;tile<g.tile_end;++tile) {
        int i=tile*32+lane, col=int(index[i]);
        if(col>=0)sum+=value[i]*x[col];
    }
    for(int d=k/2;d>0;d/=2) sum+=__shfl_down_sync(0xffffffff,sum,d,k);
    int row=g.row_start+lane/k;
    if(lane%k==0&&row<nrows)y[row_ids[row]]=sum;
}
template<int M,bool Balanced,class Index>
__device__ __forceinline__ void mixed_fixed_group(Group g,int lane,const int* row_ids,int nrows,
                                                   const Index* index,const double* value,const double* x,double* y) {
    constexpr int k=32/M;double sum=0;
    for(int tile=g.tile_start;tile<g.tile_end;++tile) {
        int i=tile*32+lane,col=int(index[i]);if(col>=0)sum+=value[i]*x[col];
    }
    for(int d=k/2;d>0;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d,k);
    int row=g.row_start+lane/k;
    if(lane%k==0&&row<nrows) {
        if constexpr(Balanced)atomicAdd(y+row_ids[row],sum);
        else y[row_ids[row]]=sum;
    }
}
template<bool Balanced,class Index>
__global__ void mixed_spmv_specialized(const Group* groups,int count,const int* row_ids,int nrows,
                                       const Index* index,const double* value,const double* x,double* y) {
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(warp>=count)return;Group g=groups[warp];
    switch(g.m) {
#define SHAPE(M) case M: mixed_fixed_group<M,Balanced>(g,lane,row_ids,nrows,index,value,x,y);break
        SHAPE(1);SHAPE(2);SHAPE(4);SHAPE(8);SHAPE(16);SHAPE(32);
#undef SHAPE
    }
}
struct Mixed {
    std::vector<int> order;
    std::vector<Group> groups;
    std::array<int,6> histogram{};
    int tiles=0;
    Group* dg=nullptr;Group* dtasks4=nullptr;Group* dtasks8=nullptr;Group* dtasks16=nullptr;int ntasks4=0,ntasks8=0,ntasks16=0;
    int* drows=nullptr;void* didx=nullptr;double* dval=nullptr;
    bool small=false;
    size_t stored_bytes=0,bytes4=0,bytes8=0,bytes16=0;
    ~Mixed(){if(dg)cudaFree(dg);if(dtasks4)cudaFree(dtasks4);if(dtasks8)cudaFree(dtasks8);if(dtasks16)cudaFree(dtasks16);if(drows)cudaFree(drows);if(didx)cudaFree(didx);if(dval)cudaFree(dval);}
};
static void make_mixed(Mixed& out,const std::vector<int>& ptr,const std::vector<int>& col,
                       const std::vector<double>& val,int ncols) {
    const int n=static_cast<int>(ptr.size())-1;
    out.order.resize(n);std::iota(out.order.begin(),out.order.end(),0);
    std::stable_sort(out.order.begin(),out.order.end(),[&](int a,int b){return ptr[a+1]-ptr[a]<ptr[b+1]-ptr[b];});
    std::vector<long long> cost(size_t(n)+1,0);std::vector<unsigned char> shape(n);
    for(int i=n-1;i>=0;--i){
        long long best=std::numeric_limits<long long>::max();int best_t=0;
        for(int t=0;t<6;++t){int m=1<<t,k=32/m,end=std::min(n,i+m);
            int r=out.order[end-1],len=ptr[r+1]-ptr[r];
            long long tiles=(static_cast<long long>(len)+k-1)/k;
            long long total=tiles*32+cost[end];
            if(total<=best){best=total;best_t=t;}
        }
        cost[i]=best;shape[i]=static_cast<unsigned char>(best_t);
    }
    if(cost[0]>INT32_MAX)throw std::overflow_error("mixed packed offsets exceed int32");
    for(int i=0;i<n;){int t=shape[i],m=1<<t,k=32/m,end=std::min(n,i+m),r=out.order[end-1];
        int len=ptr[r+1]-ptr[r];int tile_count=(len+k-1)/k;
        out.groups.push_back({i,out.tiles,out.tiles+tile_count,m});out.histogram[t]++;out.tiles+=tile_count;i=end;
    }
    // Group by shape and repack in that order so neighbouring warps follow
    // the same specialized path and access neighbouring packed tiles.
    std::stable_sort(out.groups.begin(),out.groups.end(),[](const Group& a,const Group& b){return a.m<b.m;});
    int next_tile=0;for(auto& g:out.groups){int count=g.tile_end-g.tile_start;g.tile_start=next_tile;g.tile_end=next_tile+count;next_tile+=count;}
    std::vector<double> packed(size_t(out.tiles)*32,0.0);
    std::vector<int32_t> indexes(packed.size(),-1);
    for(const auto& g:out.groups){int k=32/g.m;
        for(int off=0;off<g.m&&g.row_start+off<n;++off){int r=out.order[g.row_start+off];
            for(int j=ptr[r];j<ptr[r+1];++j){int p=j-ptr[r];size_t slot=size_t(g.tile_start+p/k)*32+off*k+p%k;packed[slot]=val[j];indexes[slot]=col[j];}
        }
    }
    auto tasks=[&](int cap){std::vector<Group> result;for(const auto& g:out.groups)
        for(int first=g.tile_start;first<g.tile_end;first+=cap)result.push_back({g.row_start,first,std::min(g.tile_end,first+cap),g.m});return result;};
    auto t4=tasks(4),t8=tasks(8),t16=tasks(16);out.ntasks4=t4.size();out.ntasks8=t8.size();out.ntasks16=t16.size();
    out.small=ncols<32767;out.dg=upload(out.groups);out.dtasks4=upload(t4);out.dtasks8=upload(t8);out.dtasks16=upload(t16);out.drows=upload(out.order);out.dval=upload(packed);
    if(out.small){std::vector<int16_t> small(indexes.begin(),indexes.end());out.didx=upload(small);}
    else out.didx=upload(indexes);
    out.stored_bytes=packed.size()*(sizeof(double)+(out.small?sizeof(int16_t):sizeof(int32_t)))+
                     out.groups.size()*sizeof(Group)+out.order.size()*sizeof(int);
    out.bytes4=out.stored_bytes-out.groups.size()*sizeof(Group)+t4.size()*sizeof(Group);
    out.bytes8=out.stored_bytes-out.groups.size()*sizeof(Group)+t8.size()*sizeof(Group);
    out.bytes16=out.stored_bytes-out.groups.size()*sizeof(Group)+t16.size()*sizeof(Group);
}
struct Candidate {std::string name;AtBsr<double,int32_t> a;int mixed=0;cudaGraph_t graph{};cudaGraphExec_t exec{};cudaEvent_t begin{},end{};std::vector<double> samples;size_t bytes=0;int slots=0;};
static void free_atbsr(AtBsr<double,int32_t>& a) {
    for(void* p:{static_cast<void*>(a.blcVal),static_cast<void*>(a.blcIdx),static_cast<void*>(a.blcIdxInt16),static_cast<void*>(a.blcPtr),static_cast<void*>(a.recorder),static_cast<void*>(a.targetRowByWarp),static_cast<void*>(a.warpBlcPtr)})if(p)ck(cudaFree(p));
}
int main(int argc,char**argv) {try {
    if(argc!=5)throw std::invalid_argument("usage: mixed_tile_bench matrix.mtx seed trials order(0|1)");
    unsigned seed=std::stoul(argv[2]);int trials=std::stoi(argv[3]),order=std::stoi(argv[4]);
    if(trials<1||trials>1000||(order!=0&&order!=1))throw std::invalid_argument("invalid trials/order");
    auto host=read_mtx(argv[1]);int rows=host.nrows,cols=host.ncols,nnz=host.nnz;
    if(rows<=0||cols<=0||nnz<=0)throw std::invalid_argument("experiment requires nonempty matrix");
    std::vector<int> ptr(host.row_ptr,host.row_ptr+rows+1),col(host.col_idx,host.col_idx+nnz);
    std::vector<double> val(host.val,host.val+nnz);delete[] host.row_ptr;delete[] host.col_idx;delete[] host.val;
    CSRMatrix<double,int32_t> a{};a.nrows=rows;a.ncols=cols;a.nnz=nnz;a.row_ptr=upload(ptr);a.col_idx=upload(col);a.val=upload(val);
    std::vector<double> x(cols),y(rows);double *dx=upload(x),*dy=upload(y);
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    std::vector<Candidate> candidates(18);candidates[0].name="mixed";candidates[0].mixed=1;
    candidates[1].name="mixed_specialized";candidates[1].mixed=2;
    candidates[2].name="mixed_bal4";candidates[2].mixed=3;
    candidates[3].name="mixed_bal8";candidates[3].mixed=4;
    candidates[4].name="mixed_bal16";candidates[4].mixed=5;candidates[5].name="default";
    for(int t=0;t<6;++t)for(int b=0;b<2;++b)candidates[6+2*t+b].name="m"+std::to_string(1<<t)+"_b"+std::to_string(b);
    Mixed mixed;
    // Both mixed execution variants share exactly one representation.
    std::vector<int> build;for(int i=0;i<18;++i)if(i==0||i>=5)build.push_back(i);if(order)std::reverse(build.begin(),build.end());
    for(int i:build){auto& c=candidates[i];
        if(i==0){make_mixed(mixed,ptr,col,val,cols);c.slots=mixed.tiles*32;c.bytes=mixed.stored_bytes;}
        else {int tile=i==5?-1:(i-6)/2,bal=i==5?-1:(i-6)%2;
            CSRToAtBSRProduct(a,c.a,tile,bal);c.slots=c.a.blockCount*32;
            c.bytes=size_t(c.slots)*(sizeof(double)+(c.a.useColIdxInt16?sizeof(int16_t):sizeof(int32_t)))+size_t(rows)*sizeof(int)+size_t(c.a.brow+1)*sizeof(int)+size_t(c.a.warpCount)*sizeof(int)+size_t(c.a.warpCount+1)*sizeof(int);
        }
    }
    candidates[1].slots=candidates[0].slots;candidates[1].bytes=candidates[0].bytes;
    candidates[2].slots=candidates[3].slots=candidates[4].slots=candidates[0].slots;
    candidates[2].bytes=mixed.bytes4;candidates[3].bytes=mixed.bytes8;candidates[4].bytes=mixed.bytes16;
    for(int i=6;i<18;++i)if(candidates[0].slots>candidates[i].slots)
        throw std::logic_error("mixed row partition uses more slots than a fixed shape");
    auto launch=[&](Candidate& c){
        if(c.mixed){int blocks=(int(mixed.groups.size())+3)/4;
            if(c.mixed==1) {
                if(mixed.small)mixed_spmv<int16_t><<<blocks,128,0,stream>>>(mixed.dg,mixed.groups.size(),mixed.drows,rows,static_cast<int16_t*>(mixed.didx),mixed.dval,dx,dy);
                else mixed_spmv<int32_t><<<blocks,128,0,stream>>>(mixed.dg,mixed.groups.size(),mixed.drows,rows,static_cast<int32_t*>(mixed.didx),mixed.dval,dx,dy);
            } else if(c.mixed==2) {
                if(mixed.small)mixed_spmv_specialized<false,int16_t><<<blocks,128,0,stream>>>(mixed.dg,mixed.groups.size(),mixed.drows,rows,static_cast<int16_t*>(mixed.didx),mixed.dval,dx,dy);
                else mixed_spmv_specialized<false,int32_t><<<blocks,128,0,stream>>>(mixed.dg,mixed.groups.size(),mixed.drows,rows,static_cast<int32_t*>(mixed.didx),mixed.dval,dx,dy);
            } else {
                int n=c.mixed==3?mixed.ntasks4:(c.mixed==4?mixed.ntasks8:mixed.ntasks16);
                Group* tasks=c.mixed==3?mixed.dtasks4:(c.mixed==4?mixed.dtasks8:mixed.dtasks16);
                ck(cudaMemsetAsync(dy,0,size_t(rows)*sizeof(double),stream));
                if(n){int grid=(n+3)/4;
                    if(mixed.small)mixed_spmv_specialized<true,int16_t><<<grid,128,0,stream>>>(tasks,n,mixed.drows,rows,static_cast<int16_t*>(mixed.didx),mixed.dval,dx,dy);
                    else mixed_spmv_specialized<true,int32_t><<<grid,128,0,stream>>>(tasks,n,mixed.drows,rows,static_cast<int32_t*>(mixed.didx),mixed.dval,dx,dy);
                }
            }
            ck(cudaGetLastError());
        }else AtSpmvProduct(c.a,dx,dy,1,0,stream);
    };
    auto verify=[&](const Candidate& c){ck(cudaStreamSynchronize(stream));ck(cudaMemcpy(y.data(),dy,size_t(rows)*sizeof(double),cudaMemcpyDeviceToHost));
        for(int row=0;row<rows;++row){long double sum=0,mag=0;for(int j=ptr[row];j<ptr[row+1];++j){long double p=(long double)val[j]*x[col[j]];sum+=p;mag+=std::abs(p);}
            long double tol=8*std::numeric_limits<double>::epsilon()*(ptr[row+1]-ptr[row]+32.L)*mag+1e-12L;
            if(!std::isfinite(y[row])||std::abs(y[row]-sum)>tol)throw std::runtime_error("incorrect "+c.name+" row="+std::to_string(row));}
    };
    for(auto& c:candidates){
        for(int v=0;v<2;++v){for(int i=0;i<cols;++i)x[i]=((i*(v?31LL:23LL)+7)%113-56)/67.;
            ck(cudaMemcpyAsync(dx,x.data(),size_t(cols)*sizeof(double),cudaMemcpyHostToDevice,stream));
            ck(cudaMemsetAsync(dy,0xff,size_t(rows)*sizeof(double),stream));launch(c);verify(c);
        }
        ck(cudaEventCreate(&c.begin));ck(cudaEventCreate(&c.end));ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        ck(cudaEventRecordWithFlags(c.begin,stream,cudaEventRecordExternal));for(int i=0;i<8;++i)launch(c);
        ck(cudaEventRecordWithFlags(c.end,stream,cudaEventRecordExternal));ck(cudaStreamEndCapture(stream,&c.graph));
        ck(cudaGraphInstantiate(&c.exec,c.graph,nullptr,nullptr,0));ck(cudaMemsetAsync(dy,0xff,size_t(rows)*sizeof(double),stream));
        ck(cudaGraphLaunch(c.exec,stream));verify(c);
    }
    std::vector<int> indices(18);std::iota(indices.begin(),indices.end(),0);std::mt19937 rng(seed);
    for(int t=-3;t<trials;++t){std::shuffle(indices.begin(),indices.end(),rng);for(int i:indices){auto& c=candidates[i];
        ck(cudaGraphLaunch(c.exec,stream));ck(cudaGraphLaunch(c.exec,stream));ck(cudaStreamSynchronize(stream));float ms=0;ck(cudaEventElapsedTime(&ms,c.begin,c.end));if(t>=0)c.samples.push_back(ms*1000./8.);
    }}
    std::cout<<"candidate,trial,us,slots,stored_bytes,rows,cols,nnz,groups_m1,groups_m2,groups_m4,groups_m8,groups_m16,groups_m32\n"<<std::setprecision(10);
    for(auto& c:candidates)for(int t=0;t<trials;++t){std::cout<<c.name<<','<<t<<','<<c.samples[t]<<','<<c.slots<<','<<c.bytes<<','<<rows<<','<<cols<<','<<nnz;for(int n:mixed.histogram)std::cout<<','<<n;std::cout<<'\n';}
    for(auto& c:candidates){ck(cudaGraphExecDestroy(c.exec));ck(cudaGraphDestroy(c.graph));ck(cudaEventDestroy(c.begin));ck(cudaEventDestroy(c.end));if(!c.mixed)free_atbsr(c.a);}
    ck(cudaFree(a.row_ptr));ck(cudaFree(a.col_idx));ck(cudaFree(a.val));ck(cudaFree(dx));ck(cudaFree(dy));ck(cudaStreamDestroy(stream));return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
