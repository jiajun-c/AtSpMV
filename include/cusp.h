#include "common.h"
#include <iostream>
#include "matrixFormat.h"
int cusparse_spmvfp64(CSRMatrix<double, int32_t> &A, double *hX, double *hY, int repeat, double &avg_time, double& pre_time);