#pragma once
#include <cuda/atomic>
// Cooperative, persistent baseline. Each warp claims one row from a
// topological order. All blocks are resident, so a spinning consumer cannot
// occupy resources needed to admit a producer block. Row publication is
// device-scoped release/acquire; each lane acquires its own dependencies.
__global__ void ordered_solve(const int* ptr,const int* col,const double* val,
    const int* rows,int* next,int* ready,int n,const double* b,double* x,int fixed_width) {
    const int lane=threadIdx.x&31;
    while(true) {
        int task=0; if(lane==0) task=atomicAdd(next,1);
        task=__shfl_sync(0xffffffff,task,0); if(task>=n) return;
        int row=rows[task], begin=ptr[row], end=ptr[row+1]-1;
        int len=end-begin; int k=fixed_width?fixed_width:(len<=2?1:len<=6?4:len<=16?8:len<=64?16:32);
        double sum=0;
        if(lane<k) for(int p=begin+lane;p<end;p+=k) {
            int j=col[p]; cuda::atomic_ref<int,cuda::thread_scope_device> flag(ready[j]);
            while(flag.load(cuda::memory_order_acquire)==0) __nanosleep(64);
            sum+=val[p]*x[j];
        }
        for(int d=16;d;d/=2) sum+=__shfl_down_sync(0xffffffff,sum,d);
        if(lane==0) {
            x[row]=(b[row]-sum)/val[end];
            cuda::atomic_ref<int,cuda::thread_scope_device> flag(ready[row]);
            flag.store(1,cuda::memory_order_release);
        }
        __syncwarp();
    }
}
