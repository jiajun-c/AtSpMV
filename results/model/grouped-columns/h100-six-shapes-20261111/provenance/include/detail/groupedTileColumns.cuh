#pragma once
#include "planStorage.cuh"
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <climits>

namespace atspmv::detail {
// Within each AtBSR row group, retain three contiguous runs of tiles. Separate
// fixed-width pools remove the metadata -> column-address dependency in SpMV.
// Normal AtBSR padding uses column 0 and value 0, so every encoded column is valid.
struct GroupedTileColumns {
    Buffer ranges, bases, columns8, columns16, columns32, values, sources;
    int tiles[3]{};
    size_t column_bytes=0;
};
static __global__ void classifyGroupedColumns(const int* ptr,const int* col,int groups,int2* meta,int* counts) {
    int group=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(group>=groups)return;
    int total[3]={};
    for(int tile=ptr[group];tile<ptr[group+1];++tile) {
        // Shared balanced packings can use -1 for padding. Normal execution
        // normalizes those zero-valued slots to the standard column-0 padding.
        int c=max(0,col[tile*32+lane]),lo=c,hi=c;
        for(int d=16;d;d/=2) {
            lo=min(lo,__shfl_down_sync(0xffffffff,lo,d));
            hi=max(hi,__shfl_down_sync(0xffffffff,hi,d));
        }
        if(lane==0) {
            int kind=hi-lo<255?0:hi-lo<65535?1:2;
            meta[tile]=make_int2(lo,kind);++total[kind];
        }
    }
    if(lane==0)for(int k=0;k<3;++k) {
        counts[k*(groups+1)+group]=total[k];
        if(group==0)counts[k*(groups+1)+groups]=0;
    }
}
static __global__ void encodeGroupedColumns(const int* ptr,const int* col,const double* val,
    int groups,const int2* meta,const int* offsets,int count8,int count16,int2* ranges,int* bases,
    unsigned char* code8,unsigned short* code16,int* code32,double* values,int2* sources) {
    int group=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(group>=groups)return;
    int cursor[3];
    for(int k=0;k<3;++k) {
        cursor[k]=offsets[k*(groups+1)+group];
        if(lane==0)ranges[group*3+k]=make_int2(cursor[k],offsets[k*(groups+1)+group+1]);
    }
    for(int tile=ptr[group];tile<ptr[group+1];++tile) {
        int2 m=meta[tile];int local=cursor[m.y]++;
        int global=local+(m.y==0?0:m.y==1?count8:count8+count16);
        if(lane==0) {
            if(m.y<2)bases[local+(m.y==1?count8:0)]=m.x;
            if(sources)sources[global]=make_int2(group,tile-ptr[group]);
        }
        int c=max(0,col[tile*32+lane]);
        if(m.y==0)code8[local*32+lane]=c-m.x;
        else if(m.y==1)code16[local*32+lane]=c-m.x;
        else code32[local*32+lane]=c;
        values[global*32+lane]=val[tile*32+lane];
    }
}
inline std::shared_ptr<GroupedTileColumns> groupTileColumns(const AtBsr<double,int32_t>& a,bool updates,cudaStream_t stream) {
    if(a.useBalance||a.useColIdxInt16||!a.blockCount||a.blockCount>INT_MAX/128)return {};
    auto result=std::make_shared<GroupedTileColumns>();Buffer meta,offsets;
    meta.allocate(size_t(a.blockCount)*sizeof(int2));
    offsets.allocate(size_t(a.brow+1)*3*sizeof(int));
    const int blocks=(a.brow+7)/8;
    classifyGroupedColumns<<<blocks,256,0,stream>>>(a.blcPtr,a.blcIdx,a.brow,meta.get<int2>(),offsets.get<int>());
    planStorageCheck(cudaGetLastError());
    for(int k=0;k<3;++k) {
        int* p=offsets.get<int>()+k*(a.brow+1);
        thrust::exclusive_scan(thrust::cuda::par.on(stream),p,p+a.brow+1,p);
        planStorageCheck(cudaMemcpyAsync(&result->tiles[k],p+a.brow,sizeof(int),cudaMemcpyDeviceToHost,stream));
    }
    planStorageCheck(cudaStreamSynchronize(stream));
    const auto* n=result->tiles;
    result->column_bytes=size_t(a.brow)*3*sizeof(int2)+size_t(n[0]+n[1])*sizeof(int)
        +size_t(n[0])*32+size_t(n[1])*64+size_t(n[2])*128;
    if(result->column_bytes>=size_t(a.blockCount)*32*sizeof(int))return {};
    result->ranges.allocate(size_t(a.brow)*3*sizeof(int2));
    result->bases.allocate(size_t(n[0]+n[1])*sizeof(int));
    result->columns8.allocate(size_t(n[0])*32);result->columns16.allocate(size_t(n[1])*64);result->columns32.allocate(size_t(n[2])*128);
    result->values.allocate(size_t(a.blockCount)*32*sizeof(double));
    if(updates)result->sources.allocate(size_t(a.blockCount)*sizeof(int2));
    encodeGroupedColumns<<<blocks,256,0,stream>>>(a.blcPtr,a.blcIdx,a.blcVal,a.brow,meta.get<int2>(),offsets.get<int>(),n[0],n[1],
        result->ranges.get<int2>(),result->bases.get<int>(),result->columns8.get<unsigned char>(),result->columns16.get<unsigned short>(),
        result->columns32.get<int>(),result->values.get<double>(),result->sources.get<int2>());
    planStorageCheck(cudaGetLastError());planStorageCheck(cudaStreamSynchronize(stream));
    return result;
}
template<class Index,bool Base>
__device__ __forceinline__ double sumGroupedTiles(int2 range,const Index* __restrict__ columns,
    const int* __restrict__ bases,const double* __restrict__ values,const double* __restrict__ x,int lane) {
    double accum[2]={};
    for(int tile=range.x;tile<range.y;tile+=2) {
        int c[2];double a[2];
        #pragma unroll
        for(int j=0;j<2;++j) {
            bool active=tile+j<range.y;
            c[j]=active?int(columns[(tile+j)*32+lane])+(Base?bases[tile+j]:0):0;
            a[j]=active?values[(tile+j)*32+lane]:0.;
        }
        #pragma unroll
        for(int j=0;j<2;++j)accum[j]+=a[j]*(tile+j<range.y?x[c[j]]:0.);
    }
    return accum[0]+accum[1];
}
template<int M>
static __global__ void groupedTileSpmv(const int2* __restrict__ ranges,const unsigned char* c8,const int* b8,const double* a8,
    const unsigned short* c16,const int* b16,const double* a16,const int* c32,const double* a32,
    const int* __restrict__ order,int rows,const double* __restrict__ x,double* __restrict__ y) {
    int group=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    constexpr int K=32/M;
    if(group>=(static_cast<long long>(rows)+M-1)/M)return;
    double sum=sumGroupedTiles<unsigned char,true>(ranges[group*3],c8,b8,a8,x,lane);
    sum+=sumGroupedTiles<unsigned short,true>(ranges[group*3+1],c16,b16,a16,x,lane);
    sum+=sumGroupedTiles<int,false>(ranges[group*3+2],c32,nullptr,a32,x,lane);
    #pragma unroll
    for(int offset=1;offset<K;offset*=2)sum+=__shfl_down_sync(0xffffffff,sum,offset,K);
    int item=group*M+lane/K;
    if(lane%K==0&&item<rows)y[order[item]]=sum;
}
inline void launchGroupedTileSpmv(const GroupedTileColumns& g,const AtBsr<double,int32_t>& a,
    const double* x,double* y,cudaStream_t stream) {
    const int warps=a.kernelConfig.warps_per_block;
#define GROUPED(M) groupedTileSpmv<M><<<(a.brow+warps-1)/warps,warps*32,0,stream>>>(g.ranges.get<int2>(),g.columns8.get<unsigned char>(),g.bases.get<int>(),g.values.get<double>(), \
        g.columns16.get<unsigned short>(),g.bases.get<int>()+g.tiles[0],g.values.get<double>()+size_t(g.tiles[0])*32, \
        g.columns32.get<int>(),g.values.get<double>()+size_t(g.tiles[0]+g.tiles[1])*32,a.recorder,a.nrows,x,y)
    switch(a.TILE_M) {
        case 1:GROUPED(1);break;case 2:GROUPED(2);break;case 4:GROUPED(4);break;
        case 8:GROUPED(8);break;case 16:GROUPED(16);break;case 32:GROUPED(32);break;
        default:throw std::invalid_argument("invalid grouped tile shape");
    }
#undef GROUPED
}
static __global__ void updateGroupedValues(const int2* sources,int tiles,const int* ptr,const int* order,
    int rows,int m,const double* values,double* packed) {
    int tile=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(tile>=tiles)return;
    int k=32/m;
    int2 source=sources[tile];int item=source.x*m+lane/k,row=item<rows?order[item]:-1;
    int pos=source.y*k+lane%k;
    packed[tile*32+lane]=row>=0&&pos<ptr[row+1]-ptr[row]?values[ptr[row]+pos]:0.;
}
} // namespace atspmv::detail
