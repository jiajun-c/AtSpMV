#include "common.h"
#include <cuda.h>

__device__ __forceinline__ void mma_m8n8k4(double *acc, double &frag_a, double &frag_b)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"
        " { %0, %1 }, "
        " { %2 }, "
        " { %3 }, "
        " { %0, %1 };"
        : "+d"(acc[0]), "+d"(acc[1]) : "d"(frag_a), "d"(frag_b));
}

__device__ __forceinline__ void store_double_to_global(const double* a, double v)
{
    asm volatile("st.global.cs.f64 [%0], %1;" :: "l"(a), "d"(v));
}

__device__ __forceinline__ double load_double_from_global(const double* a)
{
    double r;
    asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(r) : "l"(a));
    return r;
}

__device__ __forceinline__ int load_int_from_global(const int* a)
{
    int r;
    asm volatile("ld.global.cs.s32 %0, [%1];" : "=r"(r) : "l"(a));
    return r;
}

#define WARP_SIZE 32
