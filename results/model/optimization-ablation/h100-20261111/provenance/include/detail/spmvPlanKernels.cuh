#pragma once

#include <cuda_runtime.h>

namespace atspmv::detail {

// Same row/chunk assignment as the DSM backend, compiled for the normal
// architecture. One partial per CTA/row, followed by a separate gather.
template<int Rows>
__global__ void partialRows(const int* ptr,const int* col,const double* a,
                            const double* x,double* y,double* scratch,
                            const int* longs,int nl,const int* shorts,int ns,int parts) {
    const int long_blocks=((nl+Rows-1)/Rows)*parts;
    const int lane=threadIdx.x%32;
    if(blockIdx.x>=long_blocks) {
        int t=(blockIdx.x-long_blocks)*(blockDim.x/32)+threadIdx.x/32;
        int row=t<ns?shorts[t]:0;
        double sum=0;
        if(t<ns)for(int j=ptr[row]+lane;j<ptr[row+1];j+=32)sum+=a[j]*x[col[j]];
        for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
        if(t<ns&&lane==0)y[row]=sum;
        return;
    }
    __shared__ double sums[Rows][32];
    const int width=blockDim.x/Rows, local=threadIdx.x/width;
    const int task=blockIdx.x/parts*Rows+local, part=blockIdx.x%parts;
    const int tid=threadIdx.x%width, warp=tid/32;
    const int row=task<nl?longs[task]:0;
    const int begin=task<nl?ptr[row]:0, length=task<nl?ptr[row+1]-begin:0;
    const int first=begin+(long long)length*part/parts, last=begin+(long long)length*(part+1)/parts;
    double sum=0;
    for(int j=first+tid;j<last;j+=width)sum+=a[j]*x[col[j]];
    for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
    if(lane==0)sums[local][warp]=sum;
    __syncthreads();
    if(warp==0) {
        sum=lane<width/32?sums[local][lane]:0;
        for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
        if(lane==0&&task<nl)scratch[task*parts+part]=sum;
    }
}
template<int Width>
__global__ void vectorRows(const int* ptr,const int* col,const double* a,
                           const double* x,double* y,int rows) {
    constexpr int rows_per_block=256/Width;
    const auto row64=static_cast<long long>(blockIdx.x)*rows_per_block+threadIdx.x/Width;
    const int lane=threadIdx.x%Width;
    double sum=0;
    if(row64<rows) {
        const int row=static_cast<int>(row64);
        for(long long j=static_cast<long long>(ptr[row])+lane;j<ptr[row+1];j+=Width)sum+=a[j]*x[col[j]];
    }
    for(int d=Width/2;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d,Width);
    if(lane==0&&row64<rows)y[row64]=sum;
}
__global__ void finishRows(const double* scratch,double* y,const int* rows,int n,int parts) {
    int task=(blockIdx.x*blockDim.x+threadIdx.x)/32, lane=threadIdx.x%32;
    double sum=task<n&&lane<parts?scratch[task*parts+lane]:0;
    for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
    if(task<n&&lane==0)y[rows[task]]=sum;
}
} // namespace atspmv::detail
