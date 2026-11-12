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
static __global__ void solve_shared_tiles(const int* tile_ptr,const int* tile_col,const double* tile_val,
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

// For small matrices, copy the dependency data and RHS once per solve into
// shared memory. The graph must fit in one CTA; this does not drop dependencies.
template<int W> __global__ void solve_cached_csr(const int* ptr,const int* col,const double* val,
    const int* rows,const int* offsets,int nlevels,int n,int nnz,const double* b,double* x) {
    extern __shared__ double storage[];
    double *local_x=storage,*local_b=local_x+n,*local_val=local_b+n;
    int *local_ptr=(int*)(local_val+nnz),*local_col=local_ptr+n+1,*local_rows=local_col+nnz,*local_offsets=local_rows+n;
    for(int i=threadIdx.x;i<n;i+=blockDim.x) {local_b[i]=b[i];local_rows[i]=rows[i];}
    for(int i=threadIdx.x;i<nnz;i+=blockDim.x) {local_val[i]=val[i];local_col[i]=col[i];}
    for(int i=threadIdx.x;i<n+1;i+=blockDim.x)local_ptr[i]=ptr[i];
    for(int i=threadIdx.x;i<nlevels+1;i+=blockDim.x)local_offsets[i]=offsets[i];
    __syncthreads();
    int group=threadIdx.x/W,lane=threadIdx.x%W,groups=blockDim.x/W;
    for(int level=0;level<nlevels;level++) {
        int count=local_offsets[level+1]-local_offsets[level];
        for(int item=group;item<count;item+=groups) {
            int row=local_rows[local_offsets[level]+item];double sum=0.;
            for(int p=local_ptr[row]+lane;p<local_ptr[row+1]-1;p+=W)sum+=local_val[p]*local_x[local_col[p]];
            unsigned active=__activemask();for(int d=W/2;d;d/=2)sum+=__shfl_down_sync(active,sum,d,W);
            if(lane==0)local_x[row]=(local_b[row]-sum)/local_val[local_ptr[row+1]-1];
        }
        __syncthreads();
    }
    for(int i=threadIdx.x;i<n;i+=blockDim.x)x[i]=local_x[i];
}
static __global__ void solve_cached_tiles(const int* tile_ptr,const int* tile_col,const double* tile_val,
    const int* tile_row_start,const int* tile_rows,const int* level_group,const int* widths,
    const int* csr_ptr,const double* csr_val,int nlevels,int n,int ntile,int nslot,int nrow_slots,
    const double* b,double* x) {
    extern __shared__ double storage[];
    double *local_x=storage,*local_b=local_x+n,*diag=local_b+n,*local_val=diag+n;
    int *local_ptr=(int*)(local_val+nslot),*local_col=local_ptr+ntile+1,*local_row_start=local_col+nslot;
    int *local_rows=local_row_start+ntile,*local_level=local_rows+nrow_slots,*local_width=local_level+nlevels+1;
    for(int i=threadIdx.x;i<n;i+=blockDim.x){local_b[i]=b[i];diag[i]=csr_val[csr_ptr[i+1]-1];}
    for(int i=threadIdx.x;i<nslot;i+=blockDim.x){local_val[i]=tile_val[i];local_col[i]=tile_col[i];}
    for(int i=threadIdx.x;i<ntile+1;i+=blockDim.x)local_ptr[i]=tile_ptr[i];
    for(int i=threadIdx.x;i<ntile;i+=blockDim.x)local_row_start[i]=tile_row_start[i];
    for(int i=threadIdx.x;i<nrow_slots;i+=blockDim.x)local_rows[i]=tile_rows[i];
    for(int i=threadIdx.x;i<nlevels+1;i+=blockDim.x)local_level[i]=level_group[i];
    for(int i=threadIdx.x;i<nlevels;i+=blockDim.x)local_width[i]=widths[i];
    __syncthreads();
    int lane=threadIdx.x%32,warp=threadIdx.x/32,warps=blockDim.x/32;
    for(int level=0;level<nlevels;level++) {
        int k=local_width[level];
        for(int group=local_level[level]+warp;group<local_level[level+1];group+=warps) {
            int row=local_rows[local_row_start[group]+lane/k];double sum=0.;
            for(int p=local_ptr[group]+lane;p<local_ptr[group+1];p+=32){int c=local_col[p];if(c>=0)sum+=local_val[p]*local_x[c];}
            for(int d=k/2;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d,k);
            if(lane%k==0 && row>=0)local_x[row]=(local_b[row]-sum)/diag[row];
        }
        __syncthreads();
    }
    for(int i=threadIdx.x;i<n;i+=blockDim.x)x[i]=local_x[i];
}
