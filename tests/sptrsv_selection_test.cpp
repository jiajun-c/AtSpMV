#include "sptrsvSelection.h"
#include "detail/sptrsvPacking.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
using namespace atspmv;
int main() {
    assert(selectSptrsvTileThreads(255,8)==512);
    assert(selectSptrsvTileThreads(256,8)==1024);
    assert(selectSptrsvTileThreads(1,0)==512);
    SptrsvSelectionFeatures x{1024, 16, 4096, 4096, true};
    assert(selectSptrsvBackend(x) == SptrsvBackend::CachedTile);
    x.h100_pcie=false;assert(selectSptrsvBackend(x)==SptrsvBackend::CuSparse);x.h100_pcie=true;
    x.cached_bytes=4097;assert(selectSptrsvBackend(x)==SptrsvBackend::CuSparse);x.cached_bytes=4096;
    x.levels=1;assert(selectSptrsvBackend(x)==SptrsvBackend::CuSparse);
    x.levels=8;x.rows=2049;assert(selectSptrsvBackend(x)==SptrsvBackend::CuSparse);
    x.rows=2048;assert(selectSptrsvBackend(x)==SptrsvBackend::CachedTile);
    x.rows=4001;x.levels=20;assert(selectSptrsvBackend(x)==SptrsvBackend::CuSparse);
    x.rows=1;x.levels=8;assert(selectSptrsvBackend(x)==SptrsvBackend::CuSparse);
    x.rows=1024;x.levels=16;x.cached_bytes=0;assert(selectSptrsvBackend(x)==SptrsvBackend::CuSparse);
    x.rows=128;x.levels=32;x.cached_bytes=0;x.csr_cached_bytes=4096;
    assert(selectSptrsvBackend(x)==SptrsvBackend::CachedCsr);
    x.rows=129;assert(selectSptrsvBackend(x)==SptrsvBackend::CuSparse);
    x.rows=128;x.csr_cached_bytes=4097;assert(selectSptrsvBackend(x)==SptrsvBackend::CuSparse);
    x.cached_bytes=4096;assert(selectSptrsvBackend(x)==SptrsvBackend::CachedTile);x.cached_bytes=0;
    x.levels=31;x.csr_cached_bytes=4096;assert(selectSptrsvBackend(x)==SptrsvBackend::CuSparse);
    atspmv::sptrsv_detail::Matrix m{5,{0,1,2,3,4,8},{0,1,2,3,0,1,2,4},{1,1,1,1,.1,.1,.1,1}};
    auto ordinary=atspmv::sptrsv_detail::make_levels(m);
    auto six_shapes=atspmv::sptrsv_detail::make_levels(m,true);
    assert(ordinary.width.size()==2 && ordinary.width[1]==4);
    assert(six_shapes.width.size()==2 && six_shapes.width[1]==2);
    atspmv::sptrsv_detail::Matrix wide;wide.n=12;wide.ptr.push_back(0);
    for(int i=0;i<12;++i){if(i==11)for(int j=0;j<11;++j){wide.col.push_back(j);wide.val.push_back(.1);}wide.col.push_back(i);wide.val.push_back(1);wide.ptr.push_back(wide.col.size());}
    auto usual=atspmv::sptrsv_detail::make_levels(wide);
    auto narrow=atspmv::sptrsv_detail::make_levels(wide,false,true);
    assert(usual.width[1]==8 && narrow.width[1]==32);
    // Checking CSR before estimating tile bytes must preserve the result of
    // supplying both candidate sizes at once, including capacity boundaries.
    for(int rows : {1,31,32,127,128,129,1024,4000,4001})
        for(int levels : {0,1,7,8,31,32,33,256,4000})
            for(std::size_t tile_bytes : {std::size_t(0),std::size_t(4096),std::size_t(4097)})
                for(std::size_t csr_bytes : {std::size_t(0),std::size_t(4096),std::size_t(4097)})
                    for(bool calibrated : {false,true}) {
                        SptrsvSelectionFeatures f{rows,levels,tile_bytes,4096,calibrated,csr_bytes};
                        auto direct=selectSptrsvBackend(f);f.cached_bytes=0;
                        auto staged=selectSptrsvBackend(f);
                        if(staged!=SptrsvBackend::CachedCsr){f.cached_bytes=tile_bytes;staged=selectSptrsvBackend(f);}
                        assert(staged==direct);
                    }
}
