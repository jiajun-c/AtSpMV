#include "detail/sptrsvPacking.h"
#include <algorithm>
#include <map>
#include <stdexcept>
#include <vector>
using VecI=std::vector<int>;
using VecD=std::vector<double>;
using atspmv::sptrsv_detail::Matrix;
using atspmv::sptrsv_detail::make_levels;
using atspmv::sptrsv_detail::pack_levels;
#include "../bench/sptrsv_component_cached.cuh"

int main() {
    // Three components are interleaved in original row order.
    Matrix a{6,{0,1,2,4,6,7,10},{0,1,0,2,1,3,4,0,2,5},{2,3,.1,4,.2,5,6,.3,.4,7}};
    auto packed=pack_cached_components(a);
    if(packed.meta.size()!=3 || packed.map.size()!=6)return 1;
    std::size_t max_bytes=0;for(const auto& t:packed.meta)max_bytes=std::max(max_bytes,cached_component_bytes(t));
    if(max_bytes!=packed.max_shared_bytes)return 8;
    std::vector<int> seen(6,0);
    for(const auto& t:packed.meta) {
        std::vector<std::map<int,double>> found(t.n);
        for(int l=0;l<t.levels;++l) {
            int k=packed.width[t.level_base+l];
            for(int g=packed.level_group[t.level_base+l];g<packed.level_group[t.level_base+l+1];++g) {
                int row_start=packed.group_start[t.group_base+g];
                for(int i=packed.ptr[t.ptr_base+g];i<packed.ptr[t.ptr_base+g+1];++i) {
                    int row=packed.rows[t.row_base+row_start+(i%32)/k];
                    int col=packed.col[t.slot_base+i];
                    if(col<0)continue;
                    if(row<0 || row>=t.n || col>=row || !found[row].emplace(col,packed.val[t.slot_base+i]).second)return 2;
                }
            }
        }
        for(int row=0;row<t.n;++row) {
            int original=packed.map[t.map_base+row];
            if(original<0 || original>=a.n || ++seen[original]!=1)return 3;
            if(packed.diagonal[t.map_base+row]!=a.val[a.ptr[original+1]-1])return 4;
            std::map<int,double> expected;
            for(int p=a.ptr[original];p<a.ptr[original+1]-1;++p) {
                auto begin=packed.map.begin()+t.map_base,end=begin+t.n;
                auto it=std::find(begin,end,a.col[p]);if(it==end)return 5;
                expected.emplace(int(it-begin),a.val[p]);
            }
            if(found[row]!=expected)return 6;
        }
    }
    for(int count:seen)if(count!=1)return 7;
    return 0;
}
