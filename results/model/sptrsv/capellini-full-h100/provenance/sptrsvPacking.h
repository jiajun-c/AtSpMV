#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>
namespace atspmv::sptrsv_detail {
using VecI = std::vector<int>;
using VecD = std::vector<double>;
struct Matrix { int n; VecI ptr, col; VecD val; };
struct Levels { VecI rows, offsets, width; };
inline Levels make_levels(const Matrix& a, bool include_width2=false, bool narrow_width32=false) {
    VecI lev(a.n,0); int maxlev=0;
    for(int i=0;i<a.n;i++) { for(int p=a.ptr[i];p<a.ptr[i+1]-1;p++) lev[i]=std::max(lev[i],lev[a.col[p]]+1); maxlev=std::max(maxlev,lev[i]); }
    Levels l; l.offsets.assign(maxlev+2,0); for(int v:lev) l.offsets[v+1]++;
    for(int i=1;i<(int)l.offsets.size();i++) l.offsets[i]+=l.offsets[i-1];
    VecI cur=l.offsets; l.rows.resize(a.n); for(int i=0;i<a.n;i++) l.rows[cur[lev[i]]++]=i;
    for(int s=0;s<=maxlev;s++) { long long sum=0; for(int k=l.offsets[s];k<l.offsets[s+1];k++) sum+=a.ptr[l.rows[k]+1]-a.ptr[l.rows[k]]-1;
        int count=l.offsets[s+1]-l.offsets[s];double mean=double(sum)/count;
        l.width.push_back(narrow_width32 && count<=2 && mean>=8 ? 32 : mean<=2?1:(include_width2 && mean<=4)?2:mean<=6?4:mean<=16?8:mean<=64?16:32); }
    return l;
}
// AtBSR-style packing: a warp owns M independent rows, K lanes per row,
// M*K=32. Each 32-entry tile packs consecutive nonzeros from each row.
struct Packed { VecI ptr, col, rows, group_offsets; VecD val; };
inline Packed pack_levels(const Matrix& a,const Levels& levels, bool sort_by_length=false) {
    Packed z; z.ptr.push_back(0); z.group_offsets.push_back(0);
    for(size_t level=0;level<levels.width.size();++level) {
        int k=levels.width[level], m=32/k;
        VecI ordered(levels.rows.begin()+levels.offsets[level],levels.rows.begin()+levels.offsets[level+1]);
        if(sort_by_length) std::stable_sort(ordered.begin(),ordered.end(),[&](int x,int y){return (a.ptr[x+1]-a.ptr[x])<(a.ptr[y+1]-a.ptr[y]);});
        for(int start=0;start<(int)ordered.size();start+=m) {
            int maxlen=0;
            for(int r=0;r<m;r++) {int idx=start+r, row=idx<(int)ordered.size()?ordered[idx]:-1; z.rows.push_back(row); if(row>=0) maxlen=std::max(maxlen,a.ptr[row+1]-a.ptr[row]-1);}
            for(int base=0;base<maxlen;base+=k) for(int r=0;r<m;r++) {
                int row=z.rows[z.rows.size()-m+r];
                for(int lane=0;lane<k;lane++) {
                    int p=row<0?0:a.ptr[row]+base+lane; bool valid=row>=0 && p<a.ptr[row+1]-1;
                    z.col.push_back(valid?a.col[p]:-1); z.val.push_back(valid?a.val[p]:0.0);
                }
            }
            z.ptr.push_back(z.col.size());
        }
        z.group_offsets.push_back(z.ptr.size()-1);
    }
    return z;
}
} // namespace atspmv::sptrsv_detail
