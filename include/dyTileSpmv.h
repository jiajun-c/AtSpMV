#include "common.h"
#include "matrixFormat.h"
#include "optimize.h"

int dyTileSpmv(CSRMatrix<double, int32_t>&csr, TcBSR<double, int32_t>&tcbsr, Optimizer& optim, double *x, double *y, int warmup, int repeat, double &avg_time);