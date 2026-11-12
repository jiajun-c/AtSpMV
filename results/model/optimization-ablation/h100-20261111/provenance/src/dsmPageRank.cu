#include "dsmPageRank.h"
#include "tileSelector.cuh"
#include <cooperative_groups.h>

namespace atspmv {
namespace {
constexpr int Threads=256, Warps=Threads/32;
__device__ __forceinline__ unsigned clusterAddress(const double* pointer,int owner) {
    unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(pointer)),remote;
    asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(address), "r"(owner));
    return remote;
}
__device__ __forceinline__ double clusterLoad(unsigned address) {
    double value;
    asm volatile("ld.shared::cluster.f64 %0, [%1];" : "=d"(value) : "r"(address) : "memory");
    return value;
}

template<int C, bool Shared, int Rows=0>
__global__ void pageRank(const int* ptr,const int* col,const double* val,const int* dangling,
                         int n,bool has_dangling,const double* initial,double* output,
                         int iterations,double damping,double* workspace) {
    extern __shared__ double smem[];
    auto cluster=cooperative_groups::this_cluster();
    const int rank=cluster.block_rank(), lane=threadIdx.x%32, warp=threadIdx.x/32;
    const int shard=(n+C-1)/C;
    double* reduction=Shared?smem+2*shard:smem;
    for(int i=threadIdx.x;i<shard;i+=Threads) {
        int row=i*C+rank;
        if(row<n) {
            if constexpr(Shared) smem[i]=initial[row];
            else reinterpret_cast<volatile double*>(workspace)[row]=initial[row];
        }
    }
    // Reuse the first eight entries per lane across all iterations. The
    // remainder of a long row stays in CSR; nothing is truncated.
    constexpr int K=4, Slots=8;
    int starts[Rows?Rows:1], ends[Rows?Rows:1];
    unsigned cached_col[Rows?Rows:1][Slots];
    double cached_val[Rows?Rows:1][Slots];
    if constexpr(Rows>0) {
        #pragma unroll
        for(int r=0;r<Rows;++r) {
            int row=(r*(Threads/K)+threadIdx.x/K)*C+rank;
            starts[r]=row<n?ptr[row]:0; ends[r]=row<n?ptr[row+1]:0;
            #pragma unroll
            for(int j=0;j<Slots;++j) {
                int p=starts[r]+threadIdx.x%K+j*K;
                if(p<ends[r]) {
                    int index=col[p];
                    if constexpr(Shared) cached_col[r][j]=clusterAddress(smem+index/C,index&(C-1));
                    else cached_col[r][j]=index;
                } else cached_col[r][j]=~0u;
                cached_val[r][j]=p<ends[r]?val[p]:0;
            }
        }
    }
    cluster.sync();
    const double inverse_n=1.0/n;
    const double teleport=(1.0-damping)*inverse_n;
    const double dangling_scale=damping*inverse_n;
    for(int iter=0;iter<iterations;++iter) {
        const int current=iter%2,next=1-current;
        double* current_x=smem+current*shard;
        double* next_x=smem+next*shard;
        double dangling_mass=0;
        if(has_dangling) {
            double sum=0;
            for(int i=threadIdx.x;i<shard;i+=Threads) {
                int row=i*C+rank;
                if(row<n && dangling[row]) {
                    if constexpr(Shared) sum+=current_x[i];
                    else sum+=reinterpret_cast<volatile double*>(workspace)[current*n+row];
                }
            }
            for(int off=16;off;off/=2) sum+=__shfl_down_sync(0xffffffff,sum,off);
            if(lane==0) reduction[warp]=sum;
            __syncthreads();
            if(threadIdx.x==0) {
                double total=0;for(int w=0;w<Warps;++w) total+=reduction[w];
                if constexpr(Shared) reduction[0]=total;
                else reinterpret_cast<volatile double*>(workspace)[2*n+rank]=total;
            }
            cluster.sync();
            #pragma unroll
            for(int owner=0;owner<C;++owner) {
                if constexpr(Shared) dangling_mass+=clusterLoad(clusterAddress(reduction,owner));
                else dangling_mass+=reinterpret_cast<volatile double*>(workspace)[2*n+owner];
            }
            // Protect shared reduction[0] from reuse by a faster CTA next iteration.
            // The end-of-iteration cluster barrier below also protects this read.
        }
        const double base=teleport+dangling_scale*dangling_mass;
        if constexpr(Rows>0) {
            #pragma unroll
            for(int r=0;r<Rows;++r) {
                const int i=r*(Threads/K)+threadIdx.x/K,row=i*C+rank;
                double sum=0;
                double gathered[Slots];
                #pragma unroll
                for(int j=0;j<Slots;++j) {
                    unsigned index=cached_col[r][j];
                    gathered[j]=0;
                    if(index!=~0u) {
                        if constexpr(Shared) gathered[j]=clusterLoad(index+current*shard*sizeof(double));
                        else gathered[j]=reinterpret_cast<volatile double*>(workspace)[current*n+index];
                    }
                }
                #pragma unroll
                for(int j=0;j<Slots;++j) sum+=cached_val[r][j]*gathered[j];
                for(int j=starts[r]+threadIdx.x%K+Slots*K;j<ends[r];j+=K) {
                    int index=col[j];double x;
                    if constexpr(Shared) x=clusterLoad(clusterAddress(current_x+index/C,index&(C-1)));
                    else x=reinterpret_cast<volatile double*>(workspace)[current*n+index];
                    sum+=val[j]*x;
                }
                for(int off=K/2;off;off/=2) sum+=__shfl_down_sync(0xffffffff,sum,off,K);
                if(row<n&&threadIdx.x%K==0) {
                    if constexpr(Shared) next_x[i]=damping*sum+base;
                    else reinterpret_cast<volatile double*>(workspace)[next*n+row]=damping*sum+base;
                }
            }
        } else {
        for(int base_row=0;base_row<shard;base_row+=Threads/K) {
            const int i=base_row+threadIdx.x/K;
            const int row=i*C+rank;
            double sum=0;
            if(row<n) for(int j=ptr[row]+threadIdx.x%K;j<ptr[row+1];j+=K) {
                int index=col[j];double x;
                if constexpr(Shared) x=clusterLoad(clusterAddress(current_x+index/C,index&(C-1)));
                else x=reinterpret_cast<volatile double*>(workspace)[current*n+index];
                sum+=val[j]*x;
            }
            for(int off=K/2;off;off/=2) sum+=__shfl_down_sync(0xffffffff,sum,off,K);
            if(row<n && threadIdx.x%K==0) {
                if constexpr(Shared) next_x[i]=damping*sum+base;
                else reinterpret_cast<volatile double*>(workspace)[next*n+row]=damping*sum+base;
            }
        }
        }
        // Every old-vector read and next-vector write completes before swap.
        cluster.sync();
    }
    for(int i=threadIdx.x;i<shard;i+=Threads) {
        int row=i*C+rank;
        if(row<n) {
            if constexpr(Shared) output[row]=smem[(iterations%2)*shard+i];
            else output[row]=reinterpret_cast<volatile double*>(workspace)[(iterations%2)*n+row];
        }
    }
    cluster.sync();
}
template<int C,int Rows> const void* cachedKernel(XStorage storage) {
    return storage==XStorage::DistributedShared?reinterpret_cast<const void*>(pageRank<C,true,Rows>)
                                              :reinterpret_cast<const void*>(pageRank<C,false,Rows>);
}
template<int C> const void* kernel(XStorage storage,int n) {
    int rows=(n+C*(Threads/4)-1)/(C*(Threads/4));
    if(rows<=1)return cachedKernel<C,1>(storage);
    if(rows<=2)return cachedKernel<C,2>(storage);
    if(rows<=4)return cachedKernel<C,4>(storage);
    if(rows<=8)return cachedKernel<C,8>(storage);
    return cachedKernel<C,0>(storage);
}
}
struct PageRankState {
    CSRMatrix<double,int32_t> a;
    const int* dangling=nullptr;
    bool has_dangling=false;
    int clusters=0,device=-1;
    std::size_t shared=0;
    const void* kernel=nullptr;
    double* workspace=nullptr;
    ~PageRankState(){if(workspace) cudaFree(workspace);}
};
std::size_t PersistentPageRankPlan::sharedBytesPerBlock() const {return state_->shared;}
int PersistentPageRankPlan::clusterBlocks() const {return state_->clusters;}
PersistentPageRankPlan makePersistentPageRankPlan(const CSRMatrix<double,int32_t>& a,
    const int* dangling,bool has_dangling,int blocks,XStorage storage) {
    if(a.nrows<=0 || a.nrows!=a.ncols || a.nrows>100000000 ||
       (blocks!=1&&blocks!=2&&blocks!=4&&blocks!=8) || (has_dangling&&!dangling) ||
       (storage!=XStorage::Global&&storage!=XStorage::DistributedShared)) throw std::invalid_argument("invalid persistent PageRank plan");
    PersistentPageRankPlan plan;plan.state_=std::make_shared<PageRankState>();auto& s=*plan.state_;
    s.a.nrows=a.nrows;s.a.ncols=a.ncols;s.a.nnz=a.nnz;
    s.a.row_ptr=a.row_ptr;s.a.col_idx=a.col_idx;s.a.val=a.val;
    s.dangling=dangling;s.has_dangling=has_dangling;s.clusters=blocks;
    modelCudaCheck(cudaGetDevice(&s.device));int supported=0,limit=0;
    modelCudaCheck(cudaDeviceGetAttribute(&supported,cudaDevAttrClusterLaunch,s.device));
    if(!supported) throw std::runtime_error("persistent DSM PageRank requires SM90+");
    switch(blocks){case 1:s.kernel=kernel<1>(storage,a.nrows);break;case 2:s.kernel=kernel<2>(storage,a.nrows);break;
                  case 4:s.kernel=kernel<4>(storage,a.nrows);break;case 8:s.kernel=kernel<8>(storage,a.nrows);break;}
    s.shared=(Warps+(storage==XStorage::DistributedShared?2*((std::size_t(a.nrows)+blocks-1)/blocks):0))*sizeof(double);
    modelCudaCheck(cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,s.device));
    if(s.shared>std::size_t(limit)) throw std::invalid_argument("PageRank vectors exceed cluster shared capacity");
    modelCudaCheck(cudaFuncSetAttribute(s.kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,limit));
    cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={unsigned(blocks),1,1};
    cudaLaunchConfig_t config{};config.gridDim=dim3(blocks);config.blockDim=dim3(Threads);config.dynamicSmemBytes=s.shared;config.attrs=&attr;config.numAttrs=1;
    int active=0;modelCudaCheck(cudaOccupancyMaxActiveClusters(&active,s.kernel,&config));
    if(!active) throw std::runtime_error("persistent PageRank cluster cannot be scheduled");
    if(storage==XStorage::Global) modelCudaCheck(cudaMalloc(&s.workspace,(2ULL*a.nrows+blocks)*sizeof(double)));
    return plan;
}
void launchPersistentPageRank(const PersistentPageRankPlan& plan,const double* initial,double* output,
                             int iterations,double damping,cudaStream_t stream) {
    if(!plan.state_||!initial||!output||iterations<0||!std::isfinite(damping)||damping<0||damping>=1)
        throw std::invalid_argument("invalid persistent PageRank launch");
    auto& s=*plan.state_;int device;modelCudaCheck(cudaGetDevice(&device));
    if(device!=s.device) throw std::invalid_argument("persistent PageRank plan device mismatch");
    cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={unsigned(s.clusters),1,1};
    cudaLaunchConfig_t config{};config.gridDim=dim3(s.clusters);config.blockDim=dim3(Threads);
    config.dynamicSmemBytes=s.shared;config.attrs=&attr;config.numAttrs=1;config.stream=stream;
    void* args[]={&s.a.row_ptr,&s.a.col_idx,&s.a.val,&s.dangling,&s.a.nrows,&s.has_dangling,&initial,&output,&iterations,&damping,&s.workspace};
    modelCudaCheck(cudaLaunchKernelExC(&config,s.kernel,args));
}
} // namespace atspmv
