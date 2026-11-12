#pragma once
#include "planStorage.cuh"
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <climits>

namespace atspmv::detail {
// Each warp-sized tile uses byte/ushort offsets from its minimum column, or
// full int columns when the span is too large. 255/65535 encode padding.
struct CompressedTileColumns {
    Buffer meta, code;
    size_t bytes = 0;
};
static __global__ void classifyTileColumns(const int* col,int count,int2* meta,int* sizes) {
    int tile=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(tile>=count)return;
    int c=col[tile*32+lane],lo=c>=0?c:INT_MAX,hi=c;
    for(int d=16;d;d/=2) {
        lo=min(lo,__shfl_down_sync(0xffffffff,lo,d));
        hi=max(hi,__shfl_down_sync(0xffffffff,hi,d));
    }
    if(lane==0) {
        if(hi<0)lo=hi=0;
        int width=hi-lo<255?1:hi-lo<65535?2:4;
        meta[tile]=make_int2(width==1?-lo-2:width==2?lo:-1,0);
        sizes[tile]=32*width;
        if(tile==0)sizes[count]=0;
    }
}
static __global__ void encodeTileColumns(const int* col,int count,const int* offsets,
    int2* meta,unsigned char* code) {
    int tile=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(tile>=count)return;
    int base=meta[tile].x,offset=offsets[tile],c=col[tile*32+lane];
    if(lane==0)meta[tile].y=offset;
    auto* p=code+offset;
    if(base==-1)reinterpret_cast<int*>(p)[lane]=c;
    else if(base>=0)reinterpret_cast<unsigned short*>(p)[lane]=c<0?65535:c-base;
    else p[lane]=c<0?255:c+(base+2);
}
inline std::shared_ptr<CompressedTileColumns> compressTileColumns(const AtBsr<double,int32_t>& a,cudaStream_t stream) {
    if(a.useBalance||a.TILE_M!=32||a.useColIdxInt16||!a.blockCount||a.blockCount>INT_MAX/128)return {};
    auto result=std::make_shared<CompressedTileColumns>();
    result->meta.allocate(size_t(a.blockCount)*sizeof(int2));
    Buffer offsets;offsets.allocate((size_t(a.blockCount)+1)*sizeof(int));
    int blocks=(a.blockCount+7)/8;
    classifyTileColumns<<<blocks,256,0,stream>>>(a.blcIdx,a.blockCount,result->meta.get<int2>(),offsets.get<int>());
    planStorageCheck(cudaGetLastError());
    thrust::exclusive_scan(thrust::cuda::par.on(stream),offsets.get<int>(),offsets.get<int>()+a.blockCount+1,offsets.get<int>());
    int bytes=0;planStorageCheck(cudaMemcpyAsync(&bytes,offsets.get<int>()+a.blockCount,sizeof(int),cudaMemcpyDeviceToHost,stream));
    planStorageCheck(cudaStreamSynchronize(stream));
    if(size_t(bytes)+size_t(a.blockCount)*sizeof(int2)>=size_t(a.blockCount)*32*sizeof(int))return {};
    result->code.allocate(bytes);result->bytes=size_t(bytes)+size_t(a.blockCount)*sizeof(int2);
    encodeTileColumns<<<blocks,256,0,stream>>>(a.blcIdx,a.blockCount,offsets.get<int>(),result->meta.get<int2>(),result->code.get<unsigned char>());
    planStorageCheck(cudaGetLastError());planStorageCheck(cudaStreamSynchronize(stream));
    return result;
}
static __global__ void compressedTileSpmv(const int* __restrict__ ptr,const int2* __restrict__ meta,
    const unsigned char* __restrict__ code,const double* __restrict__ values,
    const int* __restrict__ order,int groups,int rows,const double* __restrict__ x,double* __restrict__ y) {
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(warp>=groups)return;
    double sums[2]={};int end=ptr[warp+1];
    for(int tile=ptr[warp];tile<end;tile+=2) {
        int c[2];double a[2];
        #pragma unroll
        for(int j=0;j<2;++j) {
            c[j]=-1;a[j]=0;
            if(tile+j<end) {
                int2 m=meta[tile+j];const auto* p=code+m.y;
                if(m.x==-1)c[j]=reinterpret_cast<const int*>(p)[lane];
                else if(m.x>=0) {int v=reinterpret_cast<const unsigned short*>(p)[lane];c[j]=v==65535?-1:m.x+v;}
                else {int v=p[lane];c[j]=v==255?-1:-(m.x+2)+v;}
                a[j]=values[(tile+j)*32+lane];
            }
        }
        #pragma unroll
        for(int j=0;j<2;++j)if(c[j]>=0)sums[j]+=a[j]*x[c[j]];
    }
    if(warp*32+lane<rows)y[order[warp*32+lane]]=sums[0]+sums[1];
}
} // namespace atspmv::detail
