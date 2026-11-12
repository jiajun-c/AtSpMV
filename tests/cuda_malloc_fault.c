// Test-only LD_PRELOAD interceptor. Fail the requested occurrence of one
// allocation size without exhausting a shared GPU's physical memory.
#define _GNU_SOURCE
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

cudaError_t cudaMalloc(void **ptr, size_t size) {
    static cudaError_t (*real_malloc)(void **, size_t);
    if (!real_malloc) real_malloc = dlsym(RTLD_NEXT, "cudaMalloc");
    const char *spec = getenv("ATSPMV_INJECT_CUDA_MALLOC");
    size_t target = 0;
    int remaining = 0;
    if (spec && sscanf(spec, "%zu:%d", &target, &remaining) == 2 && (target == 0 || target == size)) {
        if (remaining == 0) {
            unsetenv("ATSPMV_INJECT_CUDA_MALLOC"); // Fail once; allow recovery to be exercised.
            *ptr = NULL;
            if (getenv("ATSPMV_INJECT_CUDA_UNKNOWN")) {
                unsetenv("ATSPMV_INJECT_CUDA_UNKNOWN");
                return cudaErrorUnknown;
            }
            return cudaErrorMemoryAllocation;
        }
        char next[80];
        snprintf(next, sizeof(next), "%zu:%d", target, remaining-1);
        setenv("ATSPMV_INJECT_CUDA_MALLOC", next, 1);
    }
    return real_malloc(ptr, size);
}

cusparseStatus_t cusparseCreate(cusparseHandle_t *handle) {
    static cusparseStatus_t (*real_create)(cusparseHandle_t *);
    if (!real_create) real_create = dlsym(RTLD_NEXT, "cusparseCreate");
    const char *fault = getenv("ATSPMV_INJECT_CUSPARSE_CREATE");
    if (fault) {
        const int unrelated = strcmp(fault, "other") == 0;
        unsetenv("ATSPMV_INJECT_CUSPARSE_CREATE");
        *handle = NULL;
        return unrelated ? CUSPARSE_STATUS_NOT_INITIALIZED : CUSPARSE_STATUS_ALLOC_FAILED;
    }
    return real_create(handle);
}
