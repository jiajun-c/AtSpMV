#pragma once

#include "common.h"
#include <iostream>
#include "matrixFormat.h"
int cusparse_bsrfp64(CSRMatrix<double, int32_t> &A, double *hX, double *hY, int blockDim, int repeat, double &avg_time );
