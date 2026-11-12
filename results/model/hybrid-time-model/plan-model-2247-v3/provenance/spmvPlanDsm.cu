#include "detail/spmvPlanDsm.h"
#include "dsmLongRow.cuh"
namespace atspmv::detail {
void launchPlanDsm(const CSRMatrix<double,int32_t>& a, const double* x, double* y,
                  const int* longs, int nl, const int* shorts, int ns,
                  const LongRowConfig& c, cudaStream_t stream) {
#define LAUNCH(G) launchDsmLongRows<G>(a.row_ptr,a.col_idx,a.val,x,y,nl,c.parts,c.threads,stream,longs,shorts,ns)
    switch(c.rows_per_group) {
        case 1: LAUNCH(1); break;
        case 4: LAUNCH(4); break;
        case 8: LAUNCH(8); break;
    }
#undef LAUNCH
}

template<int Rows,int Width>
static void launchDsmVectorKernel(const CSRMatrix<double,int32_t>& a,const double* x,double* y,
    const int* longs,int nl,const int* shorts,int ns,const LongRowConfig& c,cudaStream_t stream) {
    const long long long_blocks=((static_cast<long long>(nl)+Rows-1)/Rows)*c.parts;
    const long long short_per_cluster=static_cast<long long>(c.threads/Width)*c.parts;
    const long long short_blocks=((static_cast<long long>(ns)+short_per_cluster-1)/short_per_cluster)*c.parts;
    if(long_blocks+short_blocks>INT_MAX)throw std::overflow_error("DSM vector grid exceeds 32-bit indexing");
    if(!long_blocks&&!short_blocks)return;
    cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={static_cast<unsigned>(c.parts),1,1};
    cudaLaunchConfig_t cfg{};cfg.blockDim=dim3(c.threads);cfg.gridDim=dim3(static_cast<unsigned>(long_blocks+short_blocks));cfg.stream=stream;cfg.attrs=&attr;cfg.numAttrs=1;
    auto err=cudaLaunchKernelEx(&cfg,long_row_detail::reduceRowsVector<Rows,Width>,a.row_ptr,a.col_idx,a.val,x,y,c.parts,longs,nl,shorts,ns);
    if(err!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(err));
}
template<int Rows>
static void launchDsmVectorWidth(const CSRMatrix<double,int32_t>& a,const double* x,double* y,const int* longs,int nl,const int* shorts,int ns,const LongRowConfig& c,int width,cudaStream_t stream) {
#define WIDTH(W) case W: launchDsmVectorKernel<Rows,W>(a,x,y,longs,nl,shorts,ns,c,stream);break
    switch(width){WIDTH(1);WIDTH(2);WIDTH(4);WIDTH(8);WIDTH(16);WIDTH(32);default:throw std::invalid_argument("invalid DSM vector width");}
#undef WIDTH
}
void launchPlanDsmVector(const CSRMatrix<double,int32_t>& a,const double* x,double* y,const int* longs,int nl,const int* shorts,int ns,const LongRowConfig& c,int width,cudaStream_t stream) {
    switch(c.rows_per_group){case 1:launchDsmVectorWidth<1>(a,x,y,longs,nl,shorts,ns,c,width,stream);break;case 4:launchDsmVectorWidth<4>(a,x,y,longs,nl,shorts,ns,c,width,stream);break;case 8:launchDsmVectorWidth<8>(a,x,y,longs,nl,shorts,ns,c,width,stream);break;default:throw std::invalid_argument("invalid row group");}
}

template<int Rows,int Tile,typename Index>
static void launchHybridKernel(const CSRMatrix<double,int32_t>& a,const AtBsr<double,int32_t>& b,
    const double* x,double* y,const int* longs,int nl,const int* shorts,int ns,
    const LongRowConfig& c,cudaStream_t stream) {
    cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={static_cast<unsigned>(c.parts),1,1};
    cudaLaunchConfig_t cfg{};cfg.blockDim=dim3(c.threads);cfg.stream=stream;cfg.attrs=&attr;cfg.numAttrs=1;
    int short_blocks=b.nnz ? (b.brow+c.threads/32-1)/(c.threads/32) : (ns+c.threads-1)/c.threads;
    short_blocks=((short_blocks+c.parts-1)/c.parts)*c.parts;
    cfg.gridDim=dim3(((nl+Rows-1)/Rows)*c.parts+short_blocks);
    if(!cfg.gridDim.x)return;
    const Index* idx;if constexpr(sizeof(Index)==2)idx=b.blcIdxInt16;else idx=b.blcIdx;
    auto err=cudaLaunchKernelEx(&cfg,long_row_detail::fusedRows<Rows,Tile,Index>,
        a.row_ptr,a.col_idx,a.val,x,y,c.parts,longs,nl,shorts,ns,b.blcPtr,idx,b.blcVal,b.recorder,b.brow);
    if(err!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(err));
}
template<int Rows,typename Index>
static void launchHybridTile(const CSRMatrix<double,int32_t>& a,const AtBsr<double,int32_t>& b,
    const double* x,double* y,const int* longs,int nl,const int* shorts,int ns,const LongRowConfig& c,cudaStream_t stream) {
#define TILE(T) case T: launchHybridKernel<Rows,T,Index>(a,b,x,y,longs,nl,shorts,ns,c,stream); break
    switch(b.TILE_M){TILE(1);TILE(2);TILE(4);TILE(8);TILE(16);TILE(32);default:throw std::invalid_argument("invalid hybrid tile");}
#undef TILE
}
template<int Rows>
static void launchHybridIndex(const CSRMatrix<double,int32_t>& a,const AtBsr<double,int32_t>& b,
    const double* x,double* y,const int* longs,int nl,const int* shorts,int ns,const LongRowConfig& c,cudaStream_t stream) {
    if(b.useColIdxInt16)launchHybridTile<Rows,int16_t>(a,b,x,y,longs,nl,shorts,ns,c,stream);
    else launchHybridTile<Rows,int32_t>(a,b,x,y,longs,nl,shorts,ns,c,stream);
}
void launchPlanHybrid(const CSRMatrix<double,int32_t>& a,const AtBsr<double,int32_t>& b,
    const double* x,double* y,const int* longs,int nl,const int* shorts,int ns,const LongRowConfig& c,cudaStream_t stream) {
    switch(c.rows_per_group){case 1:launchHybridIndex<1>(a,b,x,y,longs,nl,shorts,ns,c,stream);break;
      case 4:launchHybridIndex<4>(a,b,x,y,longs,nl,shorts,ns,c,stream);break;
      case 8:launchHybridIndex<8>(a,b,x,y,longs,nl,shorts,ns,c,stream);break;
      default:throw std::invalid_argument("invalid hybrid row group");}
}
}
