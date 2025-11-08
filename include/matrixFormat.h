#ifndef MATRIX_FORMAT_H
#define MATRIX_FORMAT_H

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
    bool useFixLen;
    int *recorder; 
    bool useColInt16;
    VAL_TYPE *val;
    IDX_TYPE *row_ptr;
    IDX_TYPE *col_idx;
};


template <typename VAL_TYPE, typename IDX_TYPE>
class TcBSR {
private:
public:
    int OptimLevel; // 对于矩阵的优化级别
    VAL_TYPE *blcVal;
    IDX_TYPE *blcIdx; // 索引信息，在Level0下存储块的起始index，其他信息自行计算
    int nrows;
    int ncols;
    int TILE_M;
    int TILE_K;
    int nnz;
    int blockCount;
    int warpCount;
    int *recorder;

    IDX_TYPE* targetRowByWarp;
    IDX_TYPE* warpBlcPtr;
    IDX_TYPE* blcPtr;
    int brow; // 行上有多少个块
    TcBSR(/* args */) {
        recorder = NULL;
        blcVal = NULL;
        blcIdx = NULL;
        targetRowByWarp = NULL;
        warpBlcPtr = NULL;
        blcPtr = NULL;
    }
    ~TcBSR() {
    }
};

#endif

