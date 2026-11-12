#pragma once
#include <cuda/atomic>
// Persistent ready queue baseline. One warp solves each row, and releases its
// dependents. The queue contains exactly n entries; each entry is written once.
struct Dependents { VecI ptr, rows, indegree, initial; VecD values; };
Dependents make_dependents(const Matrix& a) {
    Dependents d; d.ptr.assign(a.n+1,0); d.indegree.resize(a.n);
    for(int i=0;i<a.n;i++) {
        d.indegree[i]=a.ptr[i+1]-a.ptr[i]-1;
        if(!d.indegree[i]) d.initial.push_back(i);
        for(int p=a.ptr[i];p<a.ptr[i+1]-1;p++) ++d.ptr[a.col[p]+1];
    }
    for(int i=1;i<=a.n;i++) d.ptr[i]+=d.ptr[i-1];
    VecI next=d.ptr; d.rows.resize(a.val.size()-a.n); d.values.resize(d.rows.size());
    for(int i=0;i<a.n;i++) for(int p=a.ptr[i];p<a.ptr[i+1]-1;p++) {int slot=next[a.col[p]]++;d.rows[slot]=i;d.values[slot]=a.val[p];}
    return d;
}
struct QueueSlot { int row; int ready; };
__global__ void init_queue(QueueSlot* q,const int* initial,int ninitial,int* head,int* tail) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<ninitial) {q[i].row=initial[i];q[i].ready=1;}
    if(i==0) {*head=0;*tail=ninitial;}
}
__global__ void queue_solve(const int* ptr,const int* col,const double* val,
    const int* dep_ptr,const int* dep_rows,int* indegree,QueueSlot* q,int* head,int* tail,int n,const double* b,double* x) {
    int lane=threadIdx.x%32;
    while(true) {
        int index=0; if(lane==0) index=atomicAdd(head,1);
        index=__shfl_sync(0xffffffff,index,0); if(index>=n) return;
        int row=0;
        if(lane==0) {
            cuda::atomic_ref<int,cuda::thread_scope_device> ready(q[index].ready);
            while(ready.load(cuda::memory_order_acquire)==0) { __nanosleep(64); }
            row=q[index].row;
        }
        row=__shfl_sync(0xffffffff,row,0);
        // Propagate the acquire from lane 0 to readers in the warp.
        __syncwarp();
        double sum=0; for(int p=ptr[row]+lane;p<ptr[row+1]-1;p+=32) sum+=val[p]*x[col[p]];
        for(int d=16;d;d/=2) sum+=__shfl_down_sync(0xffffffff,sum,d);
        if(lane==0) {
            x[row]=(b[row]-sum)/val[ptr[row+1]-1];
            for(int p=dep_ptr[row];p<dep_ptr[row+1];p++) {
                int next=dep_rows[p]; cuda::atomic_ref<int,cuda::thread_scope_device> count(indegree[next]);
                if(count.fetch_sub(1,cuda::memory_order_acq_rel)==1) {
                    int slot=atomicAdd(tail,1); q[slot].row=next;
                    cuda::atomic_ref<int,cuda::thread_scope_device> ready(q[slot].ready);
                    ready.store(1,cuda::memory_order_release);
                }
            }
        }
        __syncwarp();
    }
}
