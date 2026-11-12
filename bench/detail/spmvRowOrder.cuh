#pragma once
#include "spmvBenchmarkSupport.cuh"
#include "detail/planStorage.cuh"
#include "kernelConfig.h"
#include <limits>
#include <numeric>

namespace atspmv::bench {
// A controlled host packer for the row-order ablation. Both orders use exactly
// the same packing algorithm and production execution kernel; setup is untimed.
inline std::shared_ptr<atspmv::detail::PackedOwner> packRowOrder(
 const std::vector<int>& p,const std::vector<int>& col,const std::vector<double>& val,
 int columns,int tile,bool sort_rows) {
 auto owner=std::make_shared<atspmv::detail::PackedOwner>();auto& a=owner->value;
 a.nrows=p.size()-1;a.ncols=columns;a.nnz=val.size();a.TILE_M=1<<tile;a.TILE_K=32/a.TILE_M;
 a.useBalance=false;a.useColIdxInt16=columns<32767;a.kernelConfig=atspmv::kernelConfigFromEnvironment();
 a.brow=a.warpCount=(a.nrows+a.TILE_M-1)/a.TILE_M;
 std::vector<int> order(a.nrows),ptr(a.brow+1);std::iota(order.begin(),order.end(),0);
 if(sort_rows)std::stable_sort(order.begin(),order.end(),[&](int x,int y){return p[x+1]-p[x]<p[y+1]-p[y];});
 for(int g=0;g<a.brow;++g) {
  int longest=0;for(int r=g*a.TILE_M;r<std::min(a.nrows,(g+1)*a.TILE_M);++r)longest=std::max(longest,p[order[r]+1]-p[order[r]]);
  long long end=static_cast<long long>(ptr[g])+(longest+a.TILE_K-1)/a.TILE_K;
  if(end>std::numeric_limits<int>::max()/32)throw std::overflow_error("row-order packing too large");
  ptr[g+1]=end;
 }
 a.blockCount=ptr.back();std::vector<int> packed_col(size_t(a.blockCount)*32);std::vector<double> packed_val(packed_col.size());
 for(int g=0;g<a.brow;++g)for(int t=ptr[g];t<ptr[g+1];++t)for(int lane=0;lane<32;++lane) {
  int item=g*a.TILE_M+lane/a.TILE_K,pos=(t-ptr[g])*a.TILE_K+lane%a.TILE_K;
  if(item<a.nrows&&pos<p[order[item]+1]-p[order[item]]) {
   int j=p[order[item]]+pos;packed_col[size_t(t)*32+lane]=col[j];packed_val[size_t(t)*32+lane]=val[j];
  }
 }
 a.recorder=upload(order);a.blcPtr=upload(ptr);a.blcVal=upload(packed_val);
 if(a.useColIdxInt16)a.blcIdxInt16=upload(std::vector<int16_t>(packed_col.begin(),packed_col.end()));
 else a.blcIdx=upload(packed_col);
 return owner;
}
} // namespace atspmv::bench
