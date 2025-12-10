#include "common.h"
#include "matrixFormat.h"

int CSRToAtBSRProduct(CSRMatrix<double, int32_t>&csrMat, AtBsr<double, int32_t>&atbsr, int idx);
int AtSpmvProduct(AtBsr<double, int32_t>& atbsr, double* d_x, double* d_y, double alpha, double beta);