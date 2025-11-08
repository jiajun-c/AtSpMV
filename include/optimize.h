#ifndef _OPTIMIZE_H_
#define _OPTIMIZE_H_
#include "common.h"
#include "matrixFormat.h"

// 可以被分块的放到到csrForBlock中，不可以分块的放入到csrRemain中

struct Optimizer {
    bool useTile;
    bool useBalance;
    bool useFixLen;
    bool useColIdxInt16;
    bool useRowPtrInt16;
    bool useCompressTile;
    int tile_m;
    int tile_k;
    int fixLen;
};
void MutliLevelOptimize(CSRMatrix<double, int32_t>& csr,CSRMatrix<double, int32_t>&csrNonZero,TcBSR<double, int32_t>&DyBsr, Optimizer& optim);
CSRMatrix<double, int32_t> recorderCSR(CSRMatrix<double, int32_t>& csr, int TILE_M, int TILE_K);
void splitCSR(CSRMatrix<double, int32_t>& csr, CSRMatrix<double, int32_t>& csrForBlock, CSRMatrix<double, int32_t>& csrRemain, int TILE_M, int TILE_K);

#endif