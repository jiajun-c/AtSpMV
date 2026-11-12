#pragma once
#include <cuda/atomic>
// Cooperative topological push. Each predecessor accumulates into successors;
// acquire/release on the dependency counter publishes every accumulated term.
__global__ void push_solve(const int* csr_ptr,const double* csr_val,
    const int* dep_ptr,const int* dep_rows,const double* dep_val,
    const int* rows,int* next,int* remaining,double* sum,int n,const double* b,double* x,int sleep_ns) {
    int lane=threadIdx.x%32;
    while(true) {
        int item=0;if(lane==0)item=atomicAdd(next,1);
        item=__shfl_sync(0xffffffff,item,0);if(item>=n)return;
        int row=rows[item];double xi=0.;
        if(lane==0) {
            cuda::atomic_ref<int,cuda::thread_scope_device> count(remaining[row]);
            while(count.load(cuda::memory_order_acquire)!=0) {if(sleep_ns) __nanosleep(sleep_ns); }
            xi=(b[row]-sum[row])/csr_val[csr_ptr[row+1]-1]; x[row]=xi;
        }
        xi=__shfl_sync(0xffffffff,xi,0);
        for(int p=dep_ptr[row]+lane;p<dep_ptr[row+1];p+=32) {
            int dst=dep_rows[p];
            cuda::atomic_ref<double,cuda::thread_scope_device> accum(sum[dst]);
            accum.fetch_add(xi*dep_val[p],cuda::memory_order_relaxed);
            cuda::atomic_ref<int,cuda::thread_scope_device> count(remaining[dst]);
            count.fetch_sub(1,cuda::memory_order_acq_rel);
        }
        __syncwarp();
    }
}
