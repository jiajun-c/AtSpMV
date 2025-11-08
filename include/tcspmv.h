#include "common.h"
#include "matrixFormat.h"

TcBSR<double, int32_t> Level1Denser2x1(CSRMatrix<double, int32_t>&csr);
TcBSR<double, int32_t> Level1Denser1x2(CSRMatrix<double, int32_t>&csr);

void BalanceTcBSR(TcBSR<double, int32_t>& tcbsr, int warpCap, double &density);
int TcSPMV2x1Balance(TcBSR<double, int32_t>& tcbsr, double* x, double* y, int warmup, int repeat, double &avgTime);
int TcSPMV1x2Balance(TcBSR<double, int32_t>& tcbsr, double* x, double* y, int warmup, int repeat, double &avgTime);
int TcSPMV2x1(TcBSR<double, int32_t>& tcbsr, double* x, double* y, int warmup, int repeat, double &avgTime);
int TcSPMV1x2(TcBSR<double, int32_t>& tcbsr, double* x, double* y, int warmup, int repeat, double &avgTime);
TcBSR<double, int32_t> Level1Denser(CSRMatrix<double, int32_t>&csr, int tile_m, int tile_k);
TcBSR<double, int32_t> Level0ToBlock(CSRMatrix<double, int32_t>&csr, double &density);
int TcSPMV1x2Level0(TcBSR<double, int32_t>& tcbsr, double* x, double* y, int warmup, int repeat, double &avgTime);
int SpmvDyTile(TcBSR<double, int32_t>& tcbsr, double* x, double* y, int tile_m, int tile_k, int warmup, int repeat, double &avgTime);
int SpmvDyTileBal(TcBSR<double, int32_t>& tcbsr, double* x, double* y, int tile_m, int tile_k, int warmup, int repeat, double &avgTime);