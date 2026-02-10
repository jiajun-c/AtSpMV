#include "common.h"
#include "matrixFormat.h"

void Spmv(CSRMatrix<double, int32_t>&csrMat, double* x, double* y, int warmup, int repeat, double& avg_time, double& pre_time,int64_t& buffsize, int idx=-1);
