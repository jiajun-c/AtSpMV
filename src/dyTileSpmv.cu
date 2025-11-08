#include "dyTileSpmv.h"
#include "csrspmv.h"
#include "tcspmv.h"
int dyTileSpmv(CSRMatrix<double, int32_t>&csr, TcBSR<double, int32_t>&tcbsr, Optimizer& optim, double *x, double *y, int warmup, int repeat, double &avg_time) {
    if (!optim.useTile) {
        // printf("use csr\n");
        csr.useFixLen = optim.useFixLen;
        csr.FixLen = optim.fixLen;
        csr.useColInt16 = optim.useColIdxInt16;
        return spmv_fp64_warpper(csr, x, y, warmup, repeat, avg_time);
    } 
    // use time 
    if (!optim.useCompressTile) {
        return TcSPMV1x2Level0(tcbsr, x, y, warmup, repeat, avg_time);
    }

    if (optim.useBalance) {
        return SpmvDyTileBal(tcbsr, x, y, tcbsr.TILE_M, tcbsr.TILE_K, warmup, repeat, avg_time);
    }
    // printf("use non balance\n");
    return SpmvDyTile(tcbsr, x, y, tcbsr.TILE_M, tcbsr.TILE_K, warmup, repeat, avg_time);
    return -1;
}