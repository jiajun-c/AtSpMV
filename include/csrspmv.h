#include "common.h"
#include "matrixFormat.h"

int spmv_fp64_balance_warpper(CSRMatrix<double, int32_t>&csr ,double* x, double* outvalues,int warmup, int repeat, double& avg_time);
int spmv_fp64_warpper(CSRMatrix<double, int32_t>&csr ,double* x, double* outvalues, int warmup, int repeat, double& avg_time);
CSRMatrix<double, int32_t> removeZeroLine(CSRMatrix<double, int32_t>&csr);