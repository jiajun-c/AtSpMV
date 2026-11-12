#pragma once
#include <cuda/atomic>
// One topologically ordered AtBSR-style tile per warp. Each tile contains
// only same-level rows, so it cannot contain an internal dependency.
__global__ void tile_wait_solve(const int* tile_ptr,const int* tile_col,const double* tile_val,
    const int* tile_row_start,const int* tile_width,const int* tile_rows,int ntiles,int* next,int* ready,
    const int* csr_ptr,const double* csr_val,const double* b,double* x) {
    int lane=threadIdx.x%32;
    while(true) {
        int task=0;if(lane==0)task=atomicAdd(next,1);
        task=__shfl_sync(0xffffffff,task,0);if(task>=ntiles)return;
        int k=tile_width[task],row=tile_rows[tile_row_start[task]+lane/k];double sum=0.;
        for(int p=tile_ptr[task]+lane;p<tile_ptr[task+1];p+=32) {
            int j=tile_col[p];if(j>=0) {
                cuda::atomic_ref<int,cuda::thread_scope_device> flag(ready[j]);
                while(flag.load(cuda::memory_order_acquire)==0) __nanosleep(32);
                sum+=tile_val[p]*x[j];
            }
        }
        for(int d=k/2;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d,k);
        if(lane%k==0 && row>=0) {
            x[row]=(b[row]-sum)/csr_val[csr_ptr[row+1]-1];
            cuda::atomic_ref<int,cuda::thread_scope_device> flag(ready[row]);flag.store(1,cuda::memory_order_release);
        }
        __syncwarp();
    }
}
