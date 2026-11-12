#pragma once
// Block-level DAG with exact intra-block level scheduling. Included after
// Matrix, CUDA(), upload(), and launch utilities in sptrsv_level_bench.cu.
struct Blocked {
    VecI block_phase, phase_group, phase_width, group_ptr, group_row, rows;
    VecI col, block_order, wave_ptr;
    VecD val;
};

Blocked pack_blocked(const Matrix& a, int block_size) {
    Blocked z; const int nblocks=(a.n+block_size-1)/block_size;
    VecI block_level(nblocks), local_level(a.n);
    std::vector<VecI> phase_rows;
    z.block_phase.push_back(0); z.phase_group.push_back(0); z.group_ptr.push_back(0);
    for(int block=0;block<nblocks;++block) {
        int start=block*block_size, end=std::min(a.n,start+block_size), max_local=0;
        for(int row=start;row<end;++row) {
            int local=0;
            for(int p=a.ptr[row];p<a.ptr[row+1]-1;++p) {
                int j=a.col[p], owner=j/block_size;
                if(owner==block) local=std::max(local,local_level[j]+1);
                else block_level[block]=std::max(block_level[block],block_level[owner]+1);
            }
            local_level[row]=local; max_local=std::max(max_local,local);
        }
        phase_rows.assign(max_local+1,{});
        for(int row=start;row<end;++row) phase_rows[local_level[row]].push_back(row);
        for(auto& phase:phase_rows) {
            long long nnz=0; for(int row:phase) nnz+=a.ptr[row+1]-a.ptr[row]-1;
            double mean=double(nnz)/phase.size();
            int k=mean<=2?1:mean<=6?4:mean<=16?8:mean<=64?16:32, m=32/k;
            z.phase_width.push_back(k);
            for(int first=0;first<(int)phase.size();first+=m) {
                int max_len=0; z.group_row.push_back(z.rows.size());
                for(int r=0;r<m;r++) {
                    int row=first+r<(int)phase.size()?phase[first+r]:-1; z.rows.push_back(row);
                    if(row>=0) max_len=std::max(max_len,a.ptr[row+1]-a.ptr[row]-1);
                }
                for(int pos=0;pos<max_len;pos+=k) for(int r=0;r<m;r++) {
                    int row=z.rows[z.rows.size()-m+r];
                    for(int lane=0;lane<k;lane++) {
                        int p=row<0?0:a.ptr[row]+pos+lane;
                        bool valid=row>=0 && p<a.ptr[row+1]-1;
                        z.col.push_back(valid?a.col[p]:-1); z.val.push_back(valid?a.val[p]:0.);
                    }
                }
                z.group_ptr.push_back(z.col.size());
            }
            z.phase_group.push_back(z.group_row.size());
        }
        z.block_phase.push_back(z.phase_width.size());
    }
    int waves=*std::max_element(block_level.begin(),block_level.end())+1;
    z.wave_ptr.assign(waves+1,0); for(int l:block_level) ++z.wave_ptr[l+1];
    for(int i=1;i<=waves;++i) z.wave_ptr[i]+=z.wave_ptr[i-1];
    VecI next=z.wave_ptr; z.block_order.resize(nblocks);
    for(int b=0;b<nblocks;++b) z.block_order[next[block_level[b]]++]=b;
    return z;
}

__global__ void blocked_packed_kernel(const int* block_order,int wave_offset,
    const int* block_phase,const int* phase_group,const int* phase_width,
    const int* group_ptr,const int* group_row,const int* rows,const int* col,const double* val,
    const int* csr_ptr,const double* csr_val,const double* b,double* x) {
    int block=block_order[wave_offset+blockIdx.x], lane=threadIdx.x%32, warp=threadIdx.x/32;
    for(int phase=block_phase[block];phase<block_phase[block+1];++phase) {
        int k=phase_width[phase];
        for(int group=phase_group[phase]+warp;group<phase_group[phase+1];group+=blockDim.x/32) {
            int row=rows[group_row[group]+lane/k]; double sum=0.;
            for(int p=group_ptr[group]+lane;p<group_ptr[group+1];p+=32) {int j=col[p]; if(j>=0) sum+=val[p]*x[j];}
            // k is uniform throughout the warp and belongs to {1,4,8,16,32}.
            for(int d=k/2;d;d/=2) sum+=__shfl_down_sync(0xffffffff,sum,d,k);
            if(lane%k==0 && row>=0) x[row]=(b[row]-sum)/csr_val[csr_ptr[row+1]-1];
        }
        __syncthreads();
    }
}

// Connected components of the undirected dependency graph are independent.
// Each component is processed by one CTA, so no grid-wide synchronization is
// required. This is an exact partition; no matrix entries are dropped.
Blocked pack_components(const Matrix& a, int forced_width=0) {
    VecI parent(a.n); for(int i=0;i<a.n;i++) parent[i]=i;
    auto root=[&](int v) {while(parent[v]!=v) {parent[v]=parent[parent[v]];v=parent[v];} return v;};
    for(int row=0;row<a.n;row++) for(int p=a.ptr[row];p<a.ptr[row+1]-1;p++) {
        int x=root(row), y=root(a.col[p]); if(x!=y) parent[x]=y;
    }
    VecI ids(a.n,-1); std::vector<VecI> components;
    for(int row=0;row<a.n;row++) {int r=root(row); if(ids[r]<0) {ids[r]=components.size();components.emplace_back();} components[ids[r]].push_back(row);}
    Blocked z; z.block_phase.push_back(0); z.phase_group.push_back(0); z.group_ptr.push_back(0);
    VecI local_level(a.n,0);
    for(const auto& component:components) {
        int max_level=0;
        for(int row:component) {for(int p=a.ptr[row];p<a.ptr[row+1]-1;p++) local_level[row]=std::max(local_level[row],local_level[a.col[p]]+1);max_level=std::max(max_level,local_level[row]);}
        std::vector<VecI> phases(max_level+1); for(int row:component) phases[local_level[row]].push_back(row);
        for(const auto& phase:phases) {
            long long nnz=0;for(int row:phase) nnz+=a.ptr[row+1]-a.ptr[row]-1;
            double mean=double(nnz)/phase.size(); int k=forced_width?forced_width:(mean<=2?1:mean<=6?4:mean<=16?8:mean<=64?16:32),m=32/k;
            z.phase_width.push_back(k);
            for(int first=0;first<(int)phase.size();first+=m) {
                int max_len=0;z.group_row.push_back(z.rows.size());
                for(int r=0;r<m;r++) {int row=first+r<(int)phase.size()?phase[first+r]:-1;z.rows.push_back(row);if(row>=0)max_len=std::max(max_len,a.ptr[row+1]-a.ptr[row]-1);}
                for(int pos=0;pos<max_len;pos+=k) for(int r=0;r<m;r++) {
                    int row=z.rows[z.rows.size()-m+r];for(int lane=0;lane<k;lane++) {
                        int p=row<0?0:a.ptr[row]+pos+lane;bool valid=row>=0 && p<a.ptr[row+1]-1;
                        z.col.push_back(valid?a.col[p]:-1);z.val.push_back(valid?a.val[p]:0.);
                    }
                }
                z.group_ptr.push_back(z.col.size());
            }
            z.phase_group.push_back(z.group_row.size());
        }
        z.block_phase.push_back(z.phase_width.size());
    }
    z.block_order.resize(components.size()); for(int b=0;b<(int)components.size();b++) z.block_order[b]=b;
    z.wave_ptr={0,(int)components.size()}; return z;
}

// A component can be processed by one warp. Independent components still run
// in separate blocks; every dependency phase ends with a warp memory barrier.
__global__ void component_warp_kernel(const int* block_order,const int* block_phase,
    const int* phase_group,const int* phase_width,const int* group_ptr,
    const int* group_row,const int* rows,const int* col,const double* val,
    const int* csr_ptr,const double* csr_val,const double* b,double* x) {
    int block=block_order[blockIdx.x],lane=threadIdx.x;
    for(int phase=block_phase[block];phase<block_phase[block+1];++phase) {
        int k=phase_width[phase];
        for(int group=phase_group[phase];group<phase_group[phase+1];++group) {
            int row=rows[group_row[group]+lane/k]; double sum=0.;
            for(int p=group_ptr[group]+lane;p<group_ptr[group+1];p+=32) {int j=col[p];if(j>=0) sum+=val[p]*x[j];}
            for(int d=k/2;d;d/=2) sum+=__shfl_down_sync(0xffffffff,sum,d,k);
            if(lane%k==0 && row>=0) x[row]=(b[row]-sum)/csr_val[csr_ptr[row+1]-1];
        }
        __syncwarp();
    }
}
