#ifndef MATRIX_FORMAT_H
#define MATRIX_FORMAT_H

#include "kernelConfig.h"

#include "common.h"
#include <fstream>

using namespace std;


template <typename VAL_TYPE, typename IDX_TYPE>
class CSRMatrix
{
private:
    /* data */
public:
    CSRMatrix(/* args */) {
        recorder = NULL;
        val = NULL;
        row_ptr = NULL;
        col_idx = NULL;
    };
    ~CSRMatrix() {

    }
    CSRMatrix(int nnz, int nrows, int ncols) {
        val = new VAL_TYPE[nnz];
        row_ptr = new IDX_TYPE[nrows + 1];
        col_idx = new IDX_TYPE[nnz];
    }
    int nnz;
    int nrows;
    int ncols;
    int FixLen;
    int minLen;
    int maxLen;
    int nonzeroRows;
    bool useFixLen;
    double stand;
    double avg_nnz;
    int *recorder; 
    bool useColInt16;
    VAL_TYPE *val;
    IDX_TYPE *row_ptr;
    IDX_TYPE *col_idx;
};


template <typename VAL_TYPE, typename IDX_TYPE>
class AtBsr {
private:
public:
    atspmv::KernelConfig kernelConfig{};
    bool modelCalibrated = false;
    int modelResidentWarps = 0;
    double predictedTimeUs = 0.0; // SpMV kernel estimate; excludes conversion/transfers.
    int OptimLevel; // 对于矩阵的优化级别
    VAL_TYPE *blcVal;
    IDX_TYPE *blcIdx; // 索引信息，在Level0下存储块的起始index，其他信息自行计算
    int16_t *blcIdxInt16;
    int nrows;
    int ncols;
    int TILE_M;
    int TILE_K;
    int nnz;
    int blockCount;
    int warpCount;
    int *recorder;
    int *recorderCPU;
    bool useColIdxInt16;
    bool useBalance;
    IDX_TYPE* targetRowByWarp;
    IDX_TYPE* warpBlcPtr;
    IDX_TYPE* blcPtr;
    int brow; // 行上有多少个块
    AtBsr(/* args */) {
        recorder = NULL;
        blcVal = NULL;
        blcIdx = NULL;
        blcIdxInt16 = NULL;
        targetRowByWarp = NULL;
        warpBlcPtr = NULL;
        blcPtr = NULL;
        useColIdxInt16 = false;
        useBalance = false;
    }
    ~AtBsr() {
    }
};

#endif
