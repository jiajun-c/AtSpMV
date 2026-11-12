#pragma once
// Experimental one-CTA-per-independent-component cached tile solve.
// Included after Matrix, VecI, VecD and sptrsv_detail packing definitions.
struct CachedComponentMeta {
    int n, levels, tiles, slots, row_slots;
    int map_base, ptr_base, slot_base, group_base, row_base, level_base;
};
struct CachedComponents {
    std::vector<CachedComponentMeta> meta;
    VecI map, ptr, col, group_start, rows, level_group, width;
    VecD val, diagonal;
    std::size_t max_shared_bytes=0;
};
inline std::size_t cached_component_bytes(const CachedComponentMeta& t) {
    return sizeof(double)*(std::size_t(3)*t.n+t.slots)+sizeof(int)*
        (std::size_t(2)*t.tiles+t.slots+t.row_slots+std::size_t(2)*t.levels+2);
}

CachedComponents pack_cached_components(const Matrix& a) {
    VecI parent(a.n), local(a.n,-1);
    for(int i=0;i<a.n;++i)parent[i]=i;
    auto root=[&](int v){while(parent[v]!=v){parent[v]=parent[parent[v]];v=parent[v];}return v;};
    for(int i=0;i<a.n;++i)for(int p=a.ptr[i];p<a.ptr[i+1]-1;++p){int x=root(i),y=root(a.col[p]);if(x!=y)parent[x]=y;}
    VecI ids(a.n,-1);std::vector<VecI> components;
    for(int i=0;i<a.n;++i){int r=root(i);if(ids[r]<0){ids[r]=components.size();components.emplace_back();}components[ids[r]].push_back(i);}
    CachedComponents out;
    for(const auto& global:components) {
        Matrix m;m.n=global.size();m.ptr.push_back(0);
        for(int i=0;i<m.n;++i)local[global[i]]=i;
        for(int row:global) {
            for(int p=a.ptr[row];p<a.ptr[row+1];++p){m.col.push_back(local[a.col[p]]);m.val.push_back(a.val[p]);}
            m.ptr.push_back(m.col.size());
        }
        auto levels=make_levels(m);auto packed=pack_levels(m,levels);
        CachedComponentMeta t{m.n,int(levels.width.size()),int(packed.ptr.size()-1),int(packed.val.size()),int(packed.rows.size()),
            int(out.map.size()),int(out.ptr.size()),int(out.val.size()),int(out.group_start.size()),int(out.rows.size()),int(out.width.size())};
        int row_base=0;
        for(std::size_t l=0;l<levels.width.size();++l)for(int g=packed.group_offsets[l];g<packed.group_offsets[l+1];++g){out.group_start.push_back(row_base);row_base+=32/levels.width[l];}
        out.map.insert(out.map.end(),global.begin(),global.end());
        for(int row:global)out.diagonal.push_back(a.val[a.ptr[row+1]-1]);
        out.ptr.insert(out.ptr.end(),packed.ptr.begin(),packed.ptr.end());
        out.col.insert(out.col.end(),packed.col.begin(),packed.col.end());
        out.val.insert(out.val.end(),packed.val.begin(),packed.val.end());
        out.rows.insert(out.rows.end(),packed.rows.begin(),packed.rows.end());
        // Store level offsets and widths together in arrays with one extra
        // sentinel per component; t.level_base points to the offsets array.
        t.level_base=out.level_group.size();
        out.level_group.insert(out.level_group.end(),packed.group_offsets.begin(),packed.group_offsets.end());
        out.width.insert(out.width.end(),levels.width.begin(),levels.width.end());out.width.push_back(0);
        out.meta.push_back(t);
        out.max_shared_bytes=std::max(out.max_shared_bytes,cached_component_bytes(t));
    }
    return out;
}

static __global__ void solve_cached_components(const CachedComponentMeta* metadata,const int* map,
    const int* ptr,const int* col,const double* val,const double* diagonal,const int* group_start,
    const int* rows,const int* level_group,const int* width,const double* b,double* x) {
    CachedComponentMeta t=metadata[blockIdx.x];
    extern __shared__ double storage[];
    double *local_x=storage,*local_b=local_x+t.n,*diag=local_b+t.n,*local_val=diag+t.n;
    int *local_ptr=(int*)(local_val+t.slots),*local_col=local_ptr+t.tiles+1,*local_group=local_col+t.slots;
    int *local_rows=local_group+t.tiles,*local_level=local_rows+t.row_slots,*local_width=local_level+t.levels+1;
    for(int i=threadIdx.x;i<t.n;i+=blockDim.x){local_b[i]=b[map[t.map_base+i]];diag[i]=diagonal[t.map_base+i];}
    for(int i=threadIdx.x;i<t.slots;i+=blockDim.x){local_val[i]=val[t.slot_base+i];local_col[i]=col[t.slot_base+i];}
    for(int i=threadIdx.x;i<t.tiles+1;i+=blockDim.x)local_ptr[i]=ptr[t.ptr_base+i];
    for(int i=threadIdx.x;i<t.tiles;i+=blockDim.x)local_group[i]=group_start[t.group_base+i];
    for(int i=threadIdx.x;i<t.row_slots;i+=blockDim.x)local_rows[i]=rows[t.row_base+i];
    for(int i=threadIdx.x;i<t.levels+1;i+=blockDim.x)local_level[i]=level_group[t.level_base+i];
    for(int i=threadIdx.x;i<t.levels;i+=blockDim.x)local_width[i]=width[t.level_base+i];
    __syncthreads();
    int lane=threadIdx.x%32,warp=threadIdx.x/32,warps=blockDim.x/32;
    for(int level=0;level<t.levels;++level) {
        int k=local_width[level];
        for(int group=local_level[level]+warp;group<local_level[level+1];group+=warps) {
            int row=local_rows[local_group[group]+lane/k];double sum=0.;
            for(int p=local_ptr[group]+lane;p<local_ptr[group+1];p+=32){int c=local_col[p];if(c>=0)sum+=local_val[p]*local_x[c];}
            for(int d=k/2;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d,k);
            if(lane%k==0 && row>=0)local_x[row]=(local_b[row]-sum)/diag[row];
        }
        __syncthreads();
    }
    for(int i=threadIdx.x;i<t.n;i+=blockDim.x)x[map[t.map_base+i]]=local_x[i];
}
