#pragma once
// Exact lower solve, one CTA. Levels and row partitions are precomputed on the
// host; each dependency level ends with a block-wide memory barrier.
template<int W> __global__ void solve_shared_csr(const int* ptr,const int* col,const double* val,
    const int* rows,const int* offsets,int nlevels,int n,const double* b,double* x) {
    extern __shared__ double local_x[];
    int group=threadIdx.x/W,lane=threadIdx.x%W,groups=blockDim.x/W;
    for(int level=0;level<nlevels;level++) {
        int count=offsets[level+1]-offsets[level];
        for(int item=group;item<count;item+=groups) {
            int row=rows[offsets[level]+item];double sum=0.;
            for(int p=ptr[row]+lane;p<ptr[row+1]-1;p+=W)sum+=val[p]*local_x[col[p]];
            unsigned active=__activemask();
            for(int d=W/2;d;d/=2)sum+=__shfl_down_sync(active,sum,d,W);
            if(lane==0)local_x[row]=(b[row]-sum)/val[ptr[row+1]-1];
        }
        __syncthreads();
    }
    for(int i=threadIdx.x;i<n;i+=blockDim.x)x[i]=local_x[i];
}
// AtBSR-style 32-slot packed tiles grouped by dependency level.
__global__ void solve_shared_tiles(const int* tile_ptr,const int* tile_col,const double* tile_val,
    const int* tile_row_start,const int* tile_rows,const int* level_group,const int* widths,
    const int* csr_ptr,const double* csr_val,int nlevels,int n,const double* b,double* x) {
    extern __shared__ double local_x[];
    int lane=threadIdx.x%32,warp=threadIdx.x/32,warps=blockDim.x/32;
    for(int level=0;level<nlevels;level++) {
        int k=widths[level];
        for(int group=level_group[level]+warp;group<level_group[level+1];group+=warps) {
            int row=tile_rows[tile_row_start[group]+lane/k];double sum=0.;
            for(int p=tile_ptr[group]+lane;p<tile_ptr[group+1];p+=32) {int col=tile_col[p];if(col>=0)sum+=tile_val[p]*local_x[col];}
            for(int d=k/2;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d,k);
            if(lane%k==0 && row>=0)local_x[row]=(b[row]-sum)/csr_val[csr_ptr[row+1]-1];
        }
        __syncthreads();
    }
    for(int i=threadIdx.x;i<n;i+=blockDim.x)x[i]=local_x[i];
}
