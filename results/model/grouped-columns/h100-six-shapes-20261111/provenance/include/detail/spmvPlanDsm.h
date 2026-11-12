#pragma once
#include "spmvPlan.h"
namespace atspmv::detail {
// Implemented in the optional SM90 object, never called on other devices.
void launchPlanDsm(const CSRMatrix<double,int32_t>&, const double*, double*,
                   const int*, int, const int*, int, const LongRowConfig&, cudaStream_t);
void launchPlanDsmVector(const CSRMatrix<double,int32_t>&, const double*, double*,
                         const int*, int, const int*, int, const LongRowConfig&, int, cudaStream_t);
void launchPlanHybrid(const CSRMatrix<double,int32_t>&, const AtBsr<double,int32_t>&,
                      const double*, double*, const int*, int, const int*, int,
                      const LongRowConfig&, cudaStream_t);
}
