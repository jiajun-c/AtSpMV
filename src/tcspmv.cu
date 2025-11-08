#include "tcspmv.h"
#include "common.h"
#include "ptx.h"
#include <cuda.h>
#include <set>
#include <map>
#include "optimize.h"
// #define WARP_CAPACITY 8
// 两种kernel
// 1. 2x1 block
// 2. 1x2 block
CSRMatrix<double, int32_t> removeZeroLine(CSRMatrix<double, int32_t>&csr) {
    int NonZeroRow = 0;
    for (int i = 0; i < csr.nrows; i++) {
        if (csr.row_ptr[i + 1] - csr.row_ptr[i] != 0) {
            NonZeroRow++;
        }
    }
    int *rowPtrNew = (int *)malloc((NonZeroRow+1) * sizeof(int));
    rowPtrNew[0] = 0;
    int cnt = 1;
    for (int i = 0; i < csr.nrows; i++) {
        if (csr.row_ptr[i + 1] - csr.row_ptr[i] != 0) {
            rowPtrNew[cnt] = csr.row_ptr[i+1];
            cnt++;
        }
    }
    CSRMatrix<double, int32_t> csr_new;
    csr_new.nrows = NonZeroRow;
    csr_new.ncols = csr.ncols;
    csr_new.nnz = csr.nnz;
    csr_new.row_ptr = rowPtrNew;
    csr_new.col_idx = csr.col_idx;
    csr_new.val = csr.val;
    return csr_new;
}

TcBSR<double, int32_t> Level0ToBlock(CSRMatrix<double, int32_t>&csr, double& density) { 
    TcBSR<double, int32_t> tcbsr;
    int brow = (csr.nrows + 3)/4;
    int *blcPtr = (int *)malloc((brow+1) * sizeof(int));
    blcPtr[0] = 0;
    vector<int>startIdx;
    for (int i = 0; i < csr.nrows; i += 4) {
        set<int>s;
        int len = min(4, csr.nrows - i);
        for (int j = 0; j < len; j++) {
            for (int k = csr.row_ptr[i+j]; k < csr.row_ptr[i + j + 1]; k++) {
                s.insert((csr.col_idx[k])/8);
            }
        }
        blcPtr[i/4+1] = s.size();
        for (auto it = s.begin(); it != s.end(); ++it) {
            startIdx.push_back(*it);
        }
    }
    for (int i = 0; i < brow; i++) {
        blcPtr[i+1] += blcPtr[i];
    }
    int blockCount = blcPtr[brow];
    double *blockVal = (double *)malloc(blockCount * 32 * sizeof(double));
    int *blockIdx = (int *)malloc(blockCount * sizeof(int));
    for (int i = 0; i < brow; i++) {
        map<int, int>m;
        for (int j = blcPtr[i], cnt = 0; j < blcPtr[i+1]; j++) {
            m[startIdx[j]] = cnt;
            blockIdx[j] = startIdx[j];
            cnt++;
        }
        for (int j = i*4; j < min(i*4+4, csr.nrows); j++) {
            for (int k = csr.row_ptr[j]; k < csr.row_ptr[j+1]; k++) {
                int col = csr.col_idx[k];
                int modcol = col % 8;
                int modrow = j % 4;
                blockVal[m[col/8]*32 + modrow*8 + modcol] = csr.val[k];
            }
        }
    }
    density = (double)csr.nnz / (double)(blockCount * 32);
    tcbsr.warpCount = brow;
    tcbsr.TILE_K = 8;
    tcbsr.TILE_M = 4;
    tcbsr.blcIdx = blockIdx;
    tcbsr.blcVal = blockVal;
    tcbsr.blcPtr = blcPtr;
    tcbsr.nnz = csr.nnz;
    tcbsr.blockCount = blockCount;
    tcbsr.brow = brow;
    tcbsr.nrows = csr.nrows;
    tcbsr.ncols = csr.ncols;
    return tcbsr;
}

TcBSR<double, int32_t> Level1Denser(CSRMatrix<double, int32_t>&csr, int tile_m, int tile_k) { 
    TcBSR<double, int32_t> tcbsr;
    vector<pair<int, int>> nnzPerRow;
    for (int i = 0; i < csr.nrows; i++) {
        nnzPerRow.push_back(make_pair(csr.row_ptr[i + 1] - csr.row_ptr[i], i));
    }
    int32_t* recorder = (int*)malloc(csr.nrows * sizeof(int));

    sort(nnzPerRow.begin(), nnzPerRow.end());
    for (int i = 0; i < csr.nrows; i++) {
        recorder[i] = nnzPerRow[i].second;
    }
    int brow = (csr.nrows + tile_m - 1)/tile_m;
    vector<int>rowPtr(csr.nrows+1, 0);
    vector<double>values(csr.nnz, 0.0);
    vector<int32_t>colIdx(csr.nnz, -1);
    for (int i = 0; i < csr.nrows; i++) {
        int oldIdx = nnzPerRow[i].second;
        for (int j = 0; j < nnzPerRow[i].first; j++) {
            values[rowPtr[i] + j] = csr.val[csr.row_ptr[oldIdx] + j];
            colIdx[rowPtr[i] + j] = csr.col_idx[csr.row_ptr[oldIdx] + j];
        }
        rowPtr[i + 1] = rowPtr[i] + nnzPerRow[i].first;
    }
    int blockCount = 0;
    int* blcPtr = new int[brow + 1];
    memset(blcPtr, 0, (brow + 1) * sizeof(int));

    for (int i = 0; i < csr.nrows; i += tile_m) {
        int startRow = i;
        int endRow = min(i + tile_m, csr.nrows);
        int maxLen = nnzPerRow[endRow - 1].first;
        blcPtr[i / tile_m + 1] = blcPtr[i/tile_m] + (maxLen + tile_k - 1) / tile_k;
    }
    blockCount = blcPtr[brow];
    // for (int i = 0; i < brow; i++) {
    //     printf("blcPtr[%d] = %d\n", i, blcPtr[i]);
    // }
    double *blockVal = (double *)malloc(blockCount * 32 * sizeof(double));
    int32_t *blockIdx = (int32_t *)malloc(blockCount * 32 * sizeof(int32_t));
    // int32_t *blockIdx = new int32_t[blockCount * 32];
    memset(blockIdx, -1, blockCount * 32 * sizeof(int32_t));
    memset(blockVal, 0.0, blockCount * 32 * sizeof(double));

    for (int i = 0; i < csr.nrows; i += tile_m) {
        int startRow = i;
        int endRow = min(i + tile_m, csr.nrows);
        int maxLen = nnzPerRow[endRow - 1].first;
        maxLen = (maxLen + tile_k - 1) / tile_k;
        for (int j = startRow; j < endRow; j ++) {
            for (int z = rowPtr[j]; z < rowPtr[j + 1]; z++) {
                int blcIdx = blcPtr[j / tile_m] + (z - rowPtr[j]) / tile_k;
                int modk = (z - rowPtr[j]) % tile_k;
                int modm = j % tile_m;
                blockVal[blcIdx * 32 + modm*tile_k + modk] = values[z];
                blockIdx[blcIdx * 32 + modm*tile_k + modk] = colIdx[z];
            }
        }
    }
    tcbsr.nnz = csr.nnz;
    tcbsr.nrows = csr.nrows;
    tcbsr.ncols = csr.ncols;
    tcbsr.TILE_K = tile_k;
    tcbsr.TILE_M = tile_m;
    tcbsr.blockCount = blockCount;
    tcbsr.blcPtr = blcPtr;
    tcbsr.blcVal = blockVal;
    tcbsr.blcIdx = blockIdx;
    tcbsr.brow = brow;
    tcbsr.recorder = recorder;
    return tcbsr;
}

TcBSR<double, int32_t> Level1Denser1x2(CSRMatrix<double, int32_t>&csr) {
    // 2x1 block
    TcBSR<double, int32_t> tcbsr;

    vector<pair<int, int>> nnzPerRow;
    for (int i = 0; i < csr.nrows; i++) {
        nnzPerRow.push_back(make_pair(csr.row_ptr[i + 1] - csr.row_ptr[i], i));
    }
    int32_t* recorder = (int*)malloc(csr.nrows * sizeof(int));

    sort(nnzPerRow.begin(), nnzPerRow.end());
    for (int i = 0; i < csr.nrows; i++) {
        recorder[i] = nnzPerRow[i].second;
    }
    int brow = (csr.nrows + 3)/4;
    vector<int>rowPtr(csr.nrows+1, 0);
    vector<double>values(csr.nnz, 0.0);
    vector<int32_t>colIdx(csr.nnz, 0);
    for (int i = 0; i < csr.nrows; i++) {
        int oldIdx = nnzPerRow[i].second;
        for (int j = 0; j < nnzPerRow[i].first; j++) {
            values[rowPtr[i] + j] = csr.val[csr.row_ptr[oldIdx] + j];
            colIdx[rowPtr[i] + j] = csr.col_idx[csr.row_ptr[oldIdx] + j];
        }
        rowPtr[i + 1] = rowPtr[i] + nnzPerRow[i].first;
    }
#ifdef DBEUG
    printf("%d %d\n", csr.nnz, rowPtr[csr.nrows]);

    printf("brow: %d\n", brow);
#endif
    int blockCount = 0;

    int* blcPtr = (int *)malloc((brow + 1) * sizeof(int));
    memset(blcPtr, 0, (brow + 1) * sizeof(int));
    for (int i = 0; i < csr.nrows; i += 4) {
        int startRow = i;
        int endRow = min(i + 4, csr.nrows);
        int maxLen = -1;
        for (int j = startRow; j < endRow; j++) { 
            maxLen = max(maxLen, nnzPerRow[j].first);
        }
        blcPtr[i / 4 + 1] = blcPtr[i/4] + (maxLen + 7) / 8;
        blockCount += (maxLen + 7) / 8;
    }
    // for (int i = 0; i <= brow; i++) {
    //     printf("blc: %d %d\n", i, blcPtr[i]);
    // }
    double *blockVal = new double[blockCount * 32];
    int32_t *blockIdx = new int32_t[blockCount * 32];
    memset(blockIdx, 0, blockCount * 32 * sizeof(int32_t));
    int prev = 0;

    for (int i = 0; i < csr.nrows; i += 4) {
        int startRow = i;
        int endRow = min(i + 4, csr.nrows);
        int maxLen = -1;
        for (int j = startRow; j < endRow; j++) { 
            maxLen = max(maxLen, nnzPerRow[j].first);
        }
        maxLen = (maxLen + 7) / 8*8;
        int tempblockCount = (maxLen + 7) / 8;
        // double* tempblockVal = new double[tempblockCount * 32];
        vector<double>tempblockVal(tempblockCount * 32, 0);
        // memset(tempblockVal, 0.0, tempblockCount * 32);
        // int32_t* tempblockIdx = new int32_t[tempblockCount * 32];
        vector<int32_t>tempblockIdx(tempblockCount * 32, -1);
        // memset(tempblockIdx, 0, tempblockCount * 32);
        for (int j = startRow, rowIdx = 0; j < endRow; j++, rowIdx++) {
            for (int z = rowPtr[j], k = 0; z < rowPtr[j+1]; z++,k++) {
                tempblockVal[rowIdx*maxLen + k] = values[z];
                tempblockIdx[rowIdx*maxLen + k] = colIdx[z];
                // if (colIdx[z] < 0) {
                //     printf("col %d\n", colIdx[z]);
                // }
            }
        }
        for (int j = 0; j < maxLen; j += 8) {
            for (int z = 0; z < 4; z++) {
                blockVal[prev + z*4]     = tempblockVal[z*maxLen + j];
                blockVal[prev + z*4 + 1] = tempblockVal[z*maxLen + j + 1];
                blockVal[prev + z*4 + 2] = tempblockVal[z*maxLen + j + 2];
                blockVal[prev + z*4 + 3] = tempblockVal[z*maxLen + j + 3];
                blockVal[prev + z*4 + 16] = tempblockVal[z*maxLen + j + 4];
                blockVal[prev + z*4 + 17] = tempblockVal[z*maxLen + j + 5];
                blockVal[prev + z*4 + 18] = tempblockVal[z*maxLen + j + 6];
                blockVal[prev + z*4 + 19] = tempblockVal[z*maxLen + j + 7];

                blockIdx[prev + z*4]     = tempblockIdx[z*maxLen + j];
                blockIdx[prev + z*4 + 1] = tempblockIdx[z*maxLen + j + 1];
                blockIdx[prev + z*4 + 2] = tempblockIdx[z*maxLen + j + 2];
                blockIdx[prev + z*4 + 3] = tempblockIdx[z*maxLen + j + 3];
                blockIdx[prev + z*4 + 16] = tempblockIdx[z*maxLen + j + 4];
                blockIdx[prev + z*4 + 17] = tempblockIdx[z*maxLen + j + 5];
                blockIdx[prev + z*4 + 18] = tempblockIdx[z*maxLen + j + 6];
                blockIdx[prev + z*4 + 19] = tempblockIdx[z*maxLen + j + 7];
            }
            // if (prev == 0) {
            //     for (int i = 0;i < 4; i++) {
            //         for (int j = 0; j < 8; j++) {
            //             printf("%f ",blockVal[i*8+j]);
            //         }
            //         printf("\n");
            //     }
            // }
            prev += 32;
        }
        // delete[] tempblockVal;
        // delete[] tempblockIdx;
    }
    tcbsr.nnz = csr.nnz;
    tcbsr.brow = brow;
    tcbsr.blcIdx = blockIdx;
    tcbsr.blcPtr = blcPtr;
    tcbsr.blcVal = blockVal;
    tcbsr.TILE_M = 4;
    tcbsr.TILE_K = 8;
    tcbsr.ncols = csr.ncols;
    tcbsr.nrows = csr.nrows;
    tcbsr.blockCount = blockCount;
    tcbsr.recorder = recorder;
    return tcbsr;
}
// #define DBEUG

TcBSR<double, int32_t> Level1Denser2x1(CSRMatrix<double, int32_t>&csr) {
    // 2x1 block
    TcBSR<double, int32_t> tcbsr;
    vector<pair<int, int>> nnzPerRow;
    for (int i = 0; i < csr.nrows; i++) {
        nnzPerRow.push_back(make_pair(csr.row_ptr[i + 1] - csr.row_ptr[i], i));
    }
    int32_t* recorder = (int*)malloc(csr.nrows * sizeof(int));

    sort(nnzPerRow.begin(), nnzPerRow.end());
    for (int i = 0; i < csr.nrows; i++) {
        recorder[i] = nnzPerRow[i].second;
    }
    int brow = (csr.nrows + 7)/8;
    vector<int>rowPtr(csr.nrows+1, 0);
    vector<double>values(csr.nnz, 0.0);
    vector<int32_t>colIdx(csr.nnz, -1);

    for (int i = 0; i < csr.nrows; i++) {
        int oldIdx = nnzPerRow[i].second;
        // printf("%d\n", nnzPerRow[i].first);
        for (int j = 0; j < nnzPerRow[i].first; j++) {
            values[rowPtr[i] + j] = csr.val[csr.row_ptr[oldIdx] + j];
            // printf("%f\n", csr.val[csr.row_ptr[oldIdx] + j]);
            // if ( csr.val[csr.row_ptr[oldIdx] + j] < 0) {
            //     zero++;
            // }
            colIdx[rowPtr[i] + j] = csr.col_idx[csr.row_ptr[oldIdx] + j];
        }
        rowPtr[i + 1] = rowPtr[i] + nnzPerRow[i].first;
    }
    // printf("%d %d\n", nnzPerRow[csr.nrows-1].first, nnzPerRow[csr.nrows-2].first);
#ifdef DBEUG
    printf("nnz %d  zero:%d %d\n", csr.nnz,zero, rowPtr[csr.nrows]);

    printf("brow: %d\n", brow);
#endif
    int blockCount = 0;
    int* blcPtr = new int[brow + 1];
    memset(blcPtr, 0, (brow + 1) * sizeof(int));
    for (int i = 0; i < csr.nrows; i += 8) {
        int startRow = i;
        int endRow = min(i + 8, csr.nrows);
        int maxLen = -1;
        int add = 0;
        for (int j = startRow; j < endRow; j++) { 
            maxLen = max(maxLen, nnzPerRow[j].first);
            add += nnzPerRow[j].first;
        }
        // printf("maxLen: %d nnz: %d\n", maxLen*8, add);
        blcPtr[i / 8 + 1] = blcPtr[i/8] + (maxLen + 3) / 4;
        blockCount += (maxLen + 3) / 4;
    }

    double *blockVal = new double[blockCount * 32];
    int32_t *blockIdx = new int32_t[blockCount * 32];
    memset(blockIdx, 0, blockCount * 32 * sizeof(int32_t));
    int prev = 0;

    for (int i = 0; i < csr.nrows; i += 8) {
        int startRow = i;
        int endRow = min(i + 8, csr.nrows);
        int maxLen = -1;
        for (int j = startRow; j < endRow; j++) { 
            maxLen = max(maxLen, nnzPerRow[j].first);
        }
        maxLen = (maxLen + 3) / 4*4;
        int tempblockCount = (maxLen + 3) / 4;
        // double* tempblockVal = new double[tempblockCount * 32];
        vector<double>tempblockVal(tempblockCount * 32, 0);
        // memset(tempblockVal, 0.0, tempblockCount * 32);
        // int32_t* tempblockIdx = new int32_t[tempblockCount * 32];
        vector<int32_t>tempblockIdx(tempblockCount * 32, -1);
        // memset(tempblockIdx, 0, tempblockCount * 32);
        for (int j = startRow, rowIdx = 0; j < endRow; j++, rowIdx++) {
            for (int z = rowPtr[j], k = 0; z < rowPtr[j+1]; z++,k++) {
                tempblockVal[rowIdx*maxLen + k] = values[z];
                tempblockIdx[rowIdx*maxLen + k] = colIdx[z];
                if (colIdx[z] < -1) {
                    printf("col %d\n", colIdx[z]);
                }
            }
        }
        for (int j = 0; j < maxLen; j += 4) {
            for (int z = 0; z < 8; z++) {
                blockVal[prev + z*4]     = tempblockVal[z*maxLen + j];
                blockVal[prev + z*4 + 1] = tempblockVal[z*maxLen + j + 1];
                blockVal[prev + z*4 + 2] = tempblockVal[z*maxLen + j + 2];
                blockVal[prev + z*4 + 3] = tempblockVal[z*maxLen + j + 3];

                blockIdx[prev + z*4]     = tempblockIdx[z*maxLen + j];
                blockIdx[prev + z*4 + 1] = tempblockIdx[z*maxLen + j + 1];
                blockIdx[prev + z*4 + 2] = tempblockIdx[z*maxLen + j + 2];
                blockIdx[prev + z*4 + 3] = tempblockIdx[z*maxLen + j + 3];
            }
            prev += 32;
        }
        // delete[] tempblockVal;
        // delete[] tempblockIdx;
    }
    tcbsr.nnz = csr.nnz;
    tcbsr.brow = brow;
    tcbsr.blcIdx = blockIdx;
    tcbsr.blcPtr = blcPtr;
    tcbsr.blcVal = blockVal;
    tcbsr.TILE_M = 8;
    tcbsr.TILE_K = 4;
    tcbsr.ncols = csr.ncols;
    tcbsr.nrows = csr.nrows;
    tcbsr.blockCount = blockCount;
    tcbsr.recorder = recorder;
    // printf("dens %f %d %d\n", double(csr.nnz)/(blockCount*32), blockCount);
    // for (int i = 0; i < blockCount*32; i++) {
    //     if (blockIdx[i] < 0) {
    //         printf("%d %d\n",i, blockIdx[i]);
    //         break;
    //     }
    // }
    return tcbsr;
}

void BalanceTcBSR(TcBSR<double, int32_t>& tcbsr, int WARP_CAPACITY, double& density) { 
    int totalWarp = 0;
    for (int i = 0; i < tcbsr.brow; i++) {
        int blc_len = tcbsr.blcPtr[i + 1] - tcbsr.blcPtr[i];
        totalWarp += (blc_len + WARP_CAPACITY - 1) / WARP_CAPACITY;
    }
    // 每个warp的起始地址
    int *warpBlcPtr = new int[totalWarp + 1];
    warpBlcPtr[0] = 0;
    int *warpTargetRow = new int[totalWarp];
    int warpIdx = 0;
    for (int i = 0; i < tcbsr.brow; i++) {
        int blc_len = tcbsr.blcPtr[i + 1] - tcbsr.blcPtr[i];
        int warpCount = (blc_len + WARP_CAPACITY - 1) / WARP_CAPACITY;
        for (int j = 1; j <= warpCount; j++) {
            // printf("min %d\n",  min(tcbsr.blcPtr[i + 1], tcbsr.blcPtr[i] + j * WARP_CAPACITY));
            warpBlcPtr[warpIdx+1] = min(tcbsr.blcPtr[i + 1], tcbsr.blcPtr[i] + j * WARP_CAPACITY);
            warpTargetRow[warpIdx] = i*tcbsr.TILE_M;
            warpIdx++;
        }
    }
    density = (double(tcbsr.nnz)/(tcbsr.blockCount*32));
    tcbsr.warpCount  = totalWarp;
    tcbsr.warpBlcPtr = warpBlcPtr;
    tcbsr.targetRowByWarp = warpTargetRow;
}

template<typename IDX_TYPE>
__global__ void TcSPMVKernelLevel01x2CC(int32_t *blcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= warpCount) return;
    int laneId = tid & 31;
    double fragA, fragB, fragC = 0.0;
    for (int i = blcPtr[warpId]; i < blcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i];
        fragA = blcVal[i*32 + laneId];
        fragB = x[colIdx + laneId%8];
        fragC += fragA * fragB;
    }
    fragC += __shfl_down_sync(0xffffffff, fragC, 1, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 2, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 4, 32);
    if (laneId % 8 == 0) {
        int rowId = warpId + laneId/8;
        if (rowId < nrows) {
            // y[rowId] = fragC;
            store_double_to_global(&y[rowId], fragC);

            // atomicAdd(&y[rowId], fragC);
        }
    }
}

template<typename IDX_TYPE>
__global__ void TcSPMVKernel1x2BalCC(IDX_TYPE *warpTargetRow, int32_t *warpBlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= warpCount) return;
    int laneId = tid & 31;
    double fragA, fragB, fragC = 0.0;
    // if (laneId == 0 && warpId == 50) {
    //     printf("warpBlcPtr %d %d targetRow %d\n", warpBlcPtr[warpId], warpBlcPtr[warpId + 1], warpTargetRow[warpId]);
    // }
    // #pragma unroll 4
    for (int i = warpBlcPtr[warpId]; i < warpBlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;
    }
    int warpRowId = warpTargetRow[warpId];
    fragC += __shfl_down_sync(0xffffffff, fragC, 1, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 2, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 4, 32);
    if (laneId % 8 == 0) {
        int rowId = warpRowId*4 + laneId/8;
        if (rowId < nrows) {
            atomicAdd(&y[rowId], fragC);
        }
    }
}

template<typename IDX_TYPE>
__global__ void TcSPMVKernel2x16CC(int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= warpCount) return;
    int laneId = tid & 31;
    double fragA, fragB, fragC = 0.0;
    // if (laneId == 0 && warpId == 50) {
    //     printf("warpBlcPtr %d %d targetRow %d\n", warpBlcPtr[warpId], warpBlcPtr[warpId + 1], warpTargetRow[warpId]);
    // }
    // #pragma unroll 4
    // if (laneId == 0) {
    //     printf("BlcPtr %d %d\n", BlcPtr[warpId], BlcPtr[warpId + 1]);
    // }
    for (int i = BlcPtr[warpId]; i < BlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;

    }
    // printf("fragA %f fragB %f fragC %f\n", fragA, fragB, fragC);
    // int warpRowId = warpTargetRow[warpId];
    fragC += __shfl_down_sync(0xffffffff, fragC, 1, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 2, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 4, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 8, 32);
    // fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    // fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);
    if (laneId  % 16 == 0) {
        int rowId = warpId*2 + laneId/16;
        if (rowId < nrows) {
            // y[rowId] = fragC;
                store_double_to_global(&y[rowId], fragC);

            // atomicAdd(&y[rowId], fragC[0]);
        }
    }
}

template<typename IDX_TYPE>
__global__ void TcSPMVKernel16x2CC(int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= warpCount) return;
    int laneId = tid & 31;
    double fragA, fragB, fragC = 0.0;
    // if (laneId == 0 && warpId == 50) {
    //     printf("warpBlcPtr %d %d targetRow %d\n", warpBlcPtr[warpId], warpBlcPtr[warpId + 1], warpTargetRow[warpId]);
    // }
    // #pragma unroll 4
    // if (laneId == 0) {
    //     printf("BlcPtr %d %d\n", BlcPtr[warpId], BlcPtr[warpId + 1]);
    // }
    for (int i = BlcPtr[warpId]; i < BlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;

    }
    // printf("fragA %f fragB %f fragC %f\n", fragA, fragB, fragC);
    // int warpRowId = warpTargetRow[warpId];
    fragC += __shfl_down_sync(0xffffffff, fragC, 1, 32);
    // fragC += __shfl_down_sync(0xffffffff, fragC, 2, 32);
    // fragC += __shfl_down_sync(0xffffffff, fragC, 4, 32);
    // fragC += __shfl_down_sync(0xffffffff, fragC, 8, 32);
    // fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    // fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);
    if (laneId  % 2 == 0) {
        int rowId = warpId*16 + laneId%2;
        if (rowId < nrows) {
            // y[rowId] = fragC;
                store_double_to_global(&y[rowId], fragC);

            // atomicAdd(&y[rowId], fragC[0]);
        }
    }
}

template<typename IDX_TYPE>
__global__ void TcSPMVKernel16x2Bal(IDX_TYPE* warpTargetRow, int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= warpCount) return;
    int laneId = tid & 31;
    double fragA, fragB, fragC = 0.0;
    // if (laneId == 0 && warpId == 50) {
    //     printf("warpBlcPtr %d %d targetRow %d\n", warpBlcPtr[warpId], warpBlcPtr[warpId + 1], warpTargetRow[warpId]);
    // }
    // #pragma unroll 4
    // if (laneId == 0) {
    //     printf("BlcPtr %d %d\n", BlcPtr[warpId], BlcPtr[warpId + 1]);
    // }
    for (int i = BlcPtr[warpId]; i < BlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;

    }
    // printf("fragA %f fragB %f fragC %f\n", fragA, fragB, fragC);
    int warpRowId = warpTargetRow[warpId];
    fragC += __shfl_down_sync(0xffffffff, fragC, 1, 32);
    // fragC += __shfl_down_sync(0xffffffff, fragC, 2, 32);
    // fragC += __shfl_down_sync(0xffffffff, fragC, 4, 32);
    // fragC += __shfl_down_sync(0xffffffff, fragC, 8, 32);
    // fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    // fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);
    if (laneId  % 2 == 0) {
        int rowId = warpRowId + laneId/2;
        if (rowId < nrows) {
            // y[rowId] = fragC;
            atomicAdd(&y[rowId], fragC);
        }
    }
}


template<typename IDX_TYPE>
__global__ void TcSPMVKernel2x16Bal(IDX_TYPE* warpTargetRow, int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= warpCount) return;
    int laneId = tid & 31;
    double fragA, fragB, fragC = 0.0;
    // if (laneId == 0 && warpId == 50) {
    //     printf("warpBlcPtr %d %d targetRow %d\n", warpBlcPtr[warpId], warpBlcPtr[warpId + 1], warpTargetRow[warpId]);
    // }
    // #pragma unroll 4
    // if (laneId == 0) {
    //     printf("BlcPtr %d %d\n", BlcPtr[warpId], BlcPtr[warpId + 1]);
    // }
    for (int i = BlcPtr[warpId]; i < BlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;

    }
    // printf("fragA %f fragB %f fragC %f\n", fragA, fragB, fragC);
    // int warpRowId = warpTargetRow[warpId];
    fragC += __shfl_down_sync(0xffffffff, fragC, 1, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 2, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 4, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 8, 32);
    // fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    // fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);
    if (laneId  % 16 == 0) {
        int rowId = warpId*2 + laneId/16;
        if (rowId < nrows) {
            // y[rowId] = fragC;
            atomicAdd(&y[rowId], fragC);
        }
    }
}


template<typename IDX_TYPE>
__global__ void TcSPMVKernel1x2CC(int32_t *BlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= warpCount) return;
    int laneId = tid & 31;
    double fragA, fragB, fragC = 0.0;
    // if (laneId == 0 && warpId == 50) {
    //     printf("warpBlcPtr %d %d targetRow %d\n", warpBlcPtr[warpId], warpBlcPtr[warpId + 1], warpTargetRow[warpId]);
    // }
    // #pragma unroll 4
    // if (laneId == 0) {
    //     printf("BlcPtr %d %d\n", BlcPtr[warpId], BlcPtr[warpId + 1]);
    // }
    for (int i = BlcPtr[warpId]; i < BlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        // if (colIdx < -1) {
            // printf("colIdx %d\n", colIdx);
        // }
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;

    }
    // int warpRowId = warpTargetRow[warpId];
    fragC += __shfl_down_sync(0xffffffff, fragC, 1, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 2, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 4, 32);
    // fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    // fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);
    if (laneId  % 8 == 0) {
        int rowId = warpId*4 + laneId/8;
        if (rowId < nrows) {
            store_double_to_global(&y[rowId], fragC);
            // y[rowId] = fragC;
            // atomicAdd(&y[rowId], fragC[0]);
        }
    }
    // if (laneId == 4) {
    //     int rowId = warpRowId + 1;
    //     if (rowId < nrows) {
    //         atomicAdd(&y[rowId], fragC[1]);
    //     }
    // }
    // if (laneId == 9) {
    //     int rowId = warpRowId + 2;
    //     if (rowId < nrows) {
    //         atomicAdd(&y[rowId], fragC[0]);
    //     }
    // }
    // if (laneId == 13) {
    //     int rowId = warpRowId + 3;
    //     if (rowId < nrows) {
    //         atomicAdd(&y[rowId], fragC[1]);
    //     }
    // }
}

template<typename IDX_TYPE>
__global__ void TcSPMVKernel2x1CC(int32_t *blcPtr, IDX_TYPE *blcIdx, double *blcVal, int brows, int nrows,  double *x, double *y) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= brows) return;
    int laneId = tid & 31;
    double fragA, fragB, fragC = 0.0;
    // if (laneId == 0) {
    //     printf("warpID: %d  blockCount: %d\n", warpId, blcPtr[warpId + 1] - blcPtr[warpId]);
    // }
    #pragma unroll 4
    for (int i = blcPtr[warpId]; i < blcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;
    }
        // if (warpId == 0)

    // printf("colIdx %d fragA %f fragB %f fragC %f\n", 1, fragA, fragB, fragC);

    fragC += __shfl_down_sync(0xffffffff, fragC, 1, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 2, 32);
    if (laneId%4 == 0) {
        int rowId = warpId*8 + laneId/4;
        if (rowId < nrows) {
            // atomicAdd(&y[rowId], fragC[0]);
            // y[rowId] = fragC;
            store_double_to_global(&y[rowId], fragC);

        }
    }

}
template<typename IDX_TYPE>
__global__ void   TcSPMVKernel2x1(int32_t *blcPtr, IDX_TYPE *blcIdx, double *blcVal, int brows, int nrows,  double *x, double *y) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= brows) return;
    int laneId = tid & 31;
    double fragA, fragB, fragC[2] ={0};
    // if (laneId == 0) {
    //     printf("warpID: %d  blockCount: %d\n", warpId, blcPtr[warpId + 1] - blcPtr[warpId]);
    // }
    #pragma unroll 4
    for (int i = blcPtr[warpId]; i < blcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        // fragC[0] += fragA * fragB;
        mma_m8n8k4(fragC, fragA, fragB);
    }
    // fragC += __shfl_down_sync(0xffffffff, fragC, 1, 32);
    // fragC += __shfl_down_sync(0xffffffff, fragC, 2, 32);
    warpId *= 4;
    if (laneId == 0) {
        int rowId = warpId;
        if (rowId < nrows) {
            // atomicAdd(&y[rowId], fragC[0]);
            y[rowId] = fragC[0];
        }
    }
    if (laneId == 4) {
        int rowId = warpId + 1;
        if (rowId < nrows) {
            // atomicAdd(&y[rowId], fragC[1]);
            y[rowId] = fragC[1];
        }
    }
    if (laneId == 9) {
        int rowId = warpId + 2;
        if (rowId < nrows) {
            // atomicAdd(&y[rowId], fragC[0]);
            y[rowId] = fragC[0];

        }
    }
    if (laneId == 13) {
        int rowId = warpId + 3;
        if (rowId < nrows) {
            // atomicAdd(&y[rowId], fragC[1]);
            y[rowId] = fragC[1];
        }
    }
    if (laneId == 18) {
        int rowId = warpId + 4;
        if (rowId < nrows) {
            y[rowId] = fragC[0];
            // atomicAdd(&y[rowId], fragC[0]);
        }
    }
    if (laneId == 22) {
        int rowId = warpId + 5;
        if (rowId < nrows) {
            // atomicAdd(&y[rowId], fragC[1]);
            y[rowId] = fragC[1];
        }
    }
    if (laneId == 27) {
        int rowId = warpId + 6;
        if (rowId < nrows) {
            // atomicAdd(&y[rowId], fragC[0]);
            y[rowId] = fragC[0];
        }
    }
    if (laneId == 31) {
        int rowId = warpId + 7;
        if (rowId < nrows) {
            // atomicAdd(&y[rowId], fragC[1]);
            y[rowId] = fragC[1];
        }
    }
}

template<typename IDX_TYPE>
__global__ void TcSPMVKernel2x1Balance(IDX_TYPE *warpTargetRow, int32_t *warpBlcPtr, IDX_TYPE *blcIdx, double *blcVal, int warpCount, int nrows,  double *x, double *y) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid >> 5;
    if (warpId >= warpCount) return;
    int laneId = tid & 31;
    double fragA, fragB, fragC = 0.0;
    // fragC[2] ={0};
    int targetRow = warpTargetRow[warpId];
    #pragma unroll 4
    for (int i = warpBlcPtr[warpId]; i < warpBlcPtr[warpId + 1]; i++) {
        int colIdx = blcIdx[i*32 + laneId];
        fragA = colIdx == -1 ? 0.0 : blcVal[i*32 + laneId];
        fragB = colIdx == -1 ? 0.0 : x[colIdx];
        fragC += fragA * fragB;
    }
    fragC += __shfl_down_sync(0xffffffff, fragC, 1, 32);
    fragC += __shfl_down_sync(0xffffffff, fragC, 2, 32);
    if (laneId%4 == 0) {
        int rowId = targetRow + laneId/4;
        if (rowId < nrows) {
            atomicAdd(&y[rowId], fragC);
            // y[rowId] = fragC;
        }
    }
    // int warpRowId = warpTargetRow[warpId];
    // if (laneId == 0) {
    //     int rowId = warpRowId;
    //     if (rowId < nrows) {
    //         atomicAdd(&y[rowId], fragC[0]);
    //     }
    // }
    // if (laneId == 4) {
    //     int rowId = warpRowId + 1;
    //     if (rowId < nrows) {
    //         atomicAdd(&y[rowId], fragC[1]);
    //     }
    // }
    // if (laneId == 9) {
    //     int rowId = warpRowId + 2;
    //     if (rowId < nrows) {
    //         atomicAdd(&y[rowId], fragC[0]);
    //     }
    // }
    // if (laneId == 13) {
    //     int rowId = warpRowId + 3;
    //     if (rowId < nrows) {
    //         atomicAdd(&y[rowId], fragC[1]);
    //     }
    // }
    // if (laneId == 18) {
    //     int rowId = warpRowId + 4;
    //     if (rowId < nrows) {
    //         atomicAdd(&y[rowId], fragC[0]);
    //     }
    // }
    // if (laneId == 22) {
    //     int rowId = warpRowId + 5;
    //     if (rowId < nrows) {
    //         atomicAdd(&y[rowId], fragC[1]);
    //     }
    // }
    // if (laneId == 27) {
    //     int rowId = warpRowId + 6;
    //     if (rowId < nrows) {
    //         atomicAdd(&y[rowId], fragC[0]);
    //     }
    // }
    // if (laneId == 31) {
    //     int rowId = warpRowId + 7;
    //     if (rowId < nrows) {
    //         atomicAdd(&y[rowId], fragC[1]);
    //     }
    // }
}

int TcSPMV2x1Balance(TcBSR<double, int32_t>& tcbsr, double* x, double* y, int warmup, int repeat, double &avgTime) {
    double *blcVal;

#ifdef DEBUG
    printf("nrows:%d ncols:%d\n", tcbsr.nrows, tcbsr.ncols);
    printf(" tcbsr.warpCount  %d\n",  tcbsr.warpCount );
#endif
    // using indexType = int32_t;
    if (tcbsr.ncols < INT16_MAX && tcbsr.nrows < INT16_MAX)  {
        printf("use int16\n");
        int16_t *warpTargetRow, *blcIdx;
        int32_t *warpBlcPtr;

        int16_t *targetRowByWarpInt16, *blcIdxfp16;
        targetRowByWarpInt16 = (int16_t*)malloc(sizeof(int16_t)*(tcbsr.warpCount ));
        blcIdxfp16 = (int16_t*)malloc(tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int16_t));
        for (int i = 0; i < tcbsr.warpCount; i++) {
            targetRowByWarpInt16[i] = tcbsr.targetRowByWarp[i];
        }
        for (int i = 0; i < tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M; i++) {
            blcIdxfp16[i] =  tcbsr.blcIdx[i];
        }
        cudaMalloc((void**)&warpTargetRow, tcbsr.warpCount * sizeof(int16_t));
        cudaMalloc((void**)&warpBlcPtr, (tcbsr.warpCount + 1) * sizeof(int32_t));
        cudaMalloc((void**)&blcIdx, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int16_t));
        cudaMalloc((void**)&blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double));

        cudaMemcpy(warpTargetRow, targetRowByWarpInt16, tcbsr.warpCount * sizeof(int16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(warpBlcPtr, tcbsr.warpBlcPtr, (tcbsr.warpCount + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcIdx, blcIdxfp16, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcVal, tcbsr.blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double), cudaMemcpyHostToDevice);

        double* d_x, *d_y;
        cudaMalloc((void**)&d_x, tcbsr.ncols * sizeof(double));
        cudaMalloc((void**)&d_y, tcbsr.nrows * sizeof(double));

        cudaMemcpy(d_x, x, tcbsr.ncols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, tcbsr.nrows * sizeof(double), cudaMemcpyHostToDevice);
        int blockSize = 4*WARP_SIZE;
        int gridSize = (tcbsr.warpCount*32 + blockSize - 1) / blockSize;
        for (int i = 0; i < warmup; i++) {
            TcSPMVKernel2x1Balance<int16_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
            cudaDeviceSynchronize();
        }
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repeat; i++) {
            TcSPMVKernel2x1Balance<int16_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
            cudaDeviceSynchronize();
        }

        auto end = std::chrono::high_resolution_clock::now(); 
        auto elapsed = end - start;
        avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        CHECK(cudaGetLastError());
        cudaMemcpy(y, d_y, tcbsr.nrows * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(warpTargetRow);
        cudaFree(warpBlcPtr);
        cudaFree(blcIdx);
        cudaFree(blcVal);
        cudaFree(d_x);
        cudaFree(d_y);
    } else {
        int32_t *warpTargetRow, *warpBlcPtr, *blcIdx;

        cudaMalloc((void**)&warpTargetRow, tcbsr.warpCount * sizeof(int32_t));
        cudaMalloc((void**)&warpBlcPtr, (tcbsr.warpCount + 1) * sizeof(int32_t));
        cudaMalloc((void**)&blcIdx, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int32_t));
        cudaMalloc((void**)&blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double));

        cudaMemcpy(warpTargetRow, tcbsr.targetRowByWarp, tcbsr.warpCount * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(warpBlcPtr, tcbsr.warpBlcPtr, (tcbsr.warpCount + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcIdx, tcbsr.blcIdx, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcVal, tcbsr.blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double), cudaMemcpyHostToDevice);
        // for (int i =0; i < tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M ; i++) {
        //     printf("%d\n", tcbsr.blcIdx[i]);
        // }
        double* d_x, *d_y;
        cudaMalloc((void**)&d_x, tcbsr.ncols * sizeof(double));
        cudaMalloc((void**)&d_y, tcbsr.nrows * sizeof(double));

        cudaMemcpy(d_x, x, tcbsr.ncols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, tcbsr.nrows * sizeof(double), cudaMemcpyHostToDevice);
        int blockSize = 4*WARP_SIZE;
        int gridSize = (tcbsr.warpCount + 4 - 1) / 4;
        // printf("gridSize %d blockSize %d\n", gridSize, blockSize);
        for (int i = 0; i < warmup; i++) {
            TcSPMVKernel2x1Balance<int32_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
            cudaDeviceSynchronize();
        }
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repeat; i++) {
            TcSPMVKernel2x1Balance<int32_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
            cudaDeviceSynchronize();
        }

        auto end = std::chrono::high_resolution_clock::now(); 
        auto elapsed = end - start;
        avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        CHECK(cudaGetLastError());
        cudaMemcpy(y, d_y, tcbsr.nrows * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(warpTargetRow);
        cudaFree(warpBlcPtr);
        cudaFree(blcIdx);
        cudaFree(blcVal);
        cudaFree(d_x);
        cudaFree(d_y);
    }
    return 0;
}

int TcSPMV2x1(TcBSR<double, int32_t>& tcbsr, double* x, double* y, int warmup, int repeat, double &avgTime) {
    double *blcVal;

#ifdef DEBUG
    printf("nrows:%d ncols:%d\n", tcbsr.nrows, tcbsr.ncols);
    printf("TcSPMV2x1 %d\n",  tcbsr.brow );
#endif
    // using indexType = int32_t;
    if (tcbsr.ncols < INT16_MAX)  {
        int16_t *blcIdx;
        int32_t *blcPtr;
        int16_t *blcIdxfp16;
        blcIdxfp16 = (int16_t*)malloc(tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int16_t));
        for (int i = 0; i < tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M; i++) {
            blcIdxfp16[i] =  tcbsr.blcIdx[i];
        }
        cudaMalloc((void**)&blcIdx, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int16_t));
        cudaMalloc((void**)&blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double));
        cudaMalloc((void**)&blcPtr, (tcbsr.brow + 1) * sizeof(int32_t));

        cudaMemcpy(blcIdx, blcIdxfp16, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcVal, tcbsr.blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(blcPtr, tcbsr.blcPtr, (tcbsr.brow + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);

        double* d_x, *d_y;
        cudaMalloc((void**)&d_x, tcbsr.ncols * sizeof(double));
        cudaMalloc((void**)&d_y, tcbsr.nrows * sizeof(double));

        cudaMemcpy(d_x, x, tcbsr.ncols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, tcbsr.nrows * sizeof(double), cudaMemcpyHostToDevice);
        int blockSize = 4*WARP_SIZE;
        int gridSize = (tcbsr.brow + 4 - 1) / 4;
        for (int i = 0; i < warmup; i++) {
            TcSPMVKernel2x1CC<int16_t><<<gridSize, blockSize>>>(blcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
            cudaDeviceSynchronize();
        }
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repeat; i++) {
            TcSPMVKernel2x1CC<int16_t><<<gridSize, blockSize>>>(blcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
            cudaDeviceSynchronize();
        }

        auto end = std::chrono::high_resolution_clock::now(); 
        auto elapsed = end - start;
        avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        CHECK(cudaGetLastError());
        cudaMemcpy(y, d_y, tcbsr.nrows * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(blcPtr);
        cudaFree(blcIdx);
        cudaFree(blcVal);
        cudaFree(d_x);
        cudaFree(d_y);
    } else {
        int32_t *blcIdx;
        int32_t *blcPtr;

        cudaMalloc((void**)&blcIdx, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int32_t));
        cudaMalloc((void**)&blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double));
        cudaMalloc((void**)&blcPtr, (tcbsr.brow + 1) * sizeof(int32_t));
        // for (int i = 0; i < tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M; i++) {
        //     if ( tcbsr.blcIdx[i] < -1)
        //     printf("%d\n", tcbsr.blcIdx[i]);
        // }

        cudaMemcpy(blcPtr, tcbsr.blcPtr, (tcbsr.brow + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcIdx, tcbsr.blcIdx, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcVal, tcbsr.blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double), cudaMemcpyHostToDevice);
        // for (int i =0; i < tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M ; i++) {
        //     printf("%d\n", tcbsr.blcIdx[i]);
        // }
        double* d_x, *d_y;
        cudaMalloc((void**)&d_x, tcbsr.ncols * sizeof(double));
        cudaMalloc((void**)&d_y, tcbsr.nrows * sizeof(double));

        cudaMemcpy(d_x, x, tcbsr.ncols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, tcbsr.nrows * sizeof(double), cudaMemcpyHostToDevice);
        int blockSize = 4*WARP_SIZE;
        int gridSize = (tcbsr.brow + 4 - 1) / 4;
        // printf("gridSize %d blockSize %d\n", gridSize, blockSize);
        for (int i = 0; i < warmup; i++) {
            TcSPMVKernel2x1CC<int32_t><<<gridSize, blockSize>>>(blcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
            cudaDeviceSynchronize();
        }
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repeat; i++) {
            TcSPMVKernel2x1CC<int32_t><<<gridSize, blockSize>>>(blcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
            cudaDeviceSynchronize();
        }

        auto end = std::chrono::high_resolution_clock::now(); 
        auto elapsed = end - start;
        avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        CHECK(cudaGetLastError());
        cudaMemcpy(y, d_y, tcbsr.nrows * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(blcPtr);
        cudaFree(blcIdx);
        cudaFree(blcVal);
        cudaFree(d_x);
        cudaFree(d_y);
    }
    return 0;
}


int TcSPMV1x2Balance(TcBSR<double, int32_t>& tcbsr, double* x, double* y, int warmup, int repeat, double &avgTime) {
    double *blcVal;

#ifdef DEBUG
    printf("nrows:%d ncols:%d\n", tcbsr.nrows, tcbsr.ncols);
    printf(" tcbsr.warpCount  %d\n",  tcbsr.warpCount );
#endif
    // using indexType = int32_t;
    if (tcbsr.ncols < INT16_MAX && tcbsr.nrows < INT16_MAX)  {
        int16_t *warpTargetRow, *blcIdx;
        int32_t *warpBlcPtr;

        int16_t *targetRowByWarpInt16, *blcIdxfp16;
        targetRowByWarpInt16 = (int16_t*)malloc(sizeof(int16_t)*(tcbsr.warpCount ));
        blcIdxfp16 = (int16_t*)malloc(tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int16_t));
        for (int i = 0; i < tcbsr.warpCount; i++) {
            targetRowByWarpInt16[i] = tcbsr.targetRowByWarp[i];
        }
        for (int i = 0; i < tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M; i++) {
            blcIdxfp16[i] =  tcbsr.blcIdx[i];
        }
        cudaMalloc((void**)&warpTargetRow, tcbsr.warpCount * sizeof(int16_t));
        cudaMalloc((void**)&warpBlcPtr, (tcbsr.warpCount + 1) * sizeof(int32_t));
        cudaMalloc((void**)&blcIdx, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int16_t));
        cudaMalloc((void**)&blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double));

        cudaMemcpy(warpTargetRow, targetRowByWarpInt16, tcbsr.warpCount * sizeof(int16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(warpBlcPtr, tcbsr.warpBlcPtr, (tcbsr.warpCount + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcIdx, blcIdxfp16, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcVal, tcbsr.blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double), cudaMemcpyHostToDevice);

        double* d_x, *d_y;
        cudaMalloc((void**)&d_x, tcbsr.ncols * sizeof(double));
        cudaMalloc((void**)&d_y, tcbsr.nrows * sizeof(double));

        cudaMemcpy(d_x, x, tcbsr.ncols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, tcbsr.nrows * sizeof(double), cudaMemcpyHostToDevice);
        int blockSize = 4*WARP_SIZE;
        int gridSize = (tcbsr.warpCount + 4 - 1) / 4;
        for (int i = 0; i < warmup; i++) {
            TcSPMVKernel1x2BalCC<int16_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
            cudaDeviceSynchronize();
        }
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repeat; i++) {
            TcSPMVKernel1x2BalCC<int16_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
            cudaDeviceSynchronize();
        }

        auto end = std::chrono::high_resolution_clock::now(); 
        auto elapsed = end - start;
        avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        CHECK(cudaGetLastError());
        cudaMemcpy(y, d_y, tcbsr.nrows * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(warpTargetRow);
        cudaFree(warpBlcPtr);
        cudaFree(blcIdx);
        cudaFree(blcVal);
        cudaFree(d_x);
        cudaFree(d_y);
    } else {
        int32_t *warpTargetRow, *warpBlcPtr, *blcIdx;

        cudaMalloc((void**)&warpTargetRow, tcbsr.warpCount * sizeof(int32_t));
        cudaMalloc((void**)&warpBlcPtr, (tcbsr.warpCount + 1) * sizeof(int32_t));
        cudaMalloc((void**)&blcIdx, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int32_t));
        cudaMalloc((void**)&blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double));

        cudaMemcpy(warpTargetRow, tcbsr.targetRowByWarp, tcbsr.warpCount * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(warpBlcPtr, tcbsr.warpBlcPtr, (tcbsr.warpCount + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcIdx, tcbsr.blcIdx, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcVal, tcbsr.blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double), cudaMemcpyHostToDevice);
        // for (int i =0; i < tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M ; i++) {
        //     printf("%d\n", tcbsr.blcIdx[i]);
        // }
        double* d_x, *d_y;
        cudaMalloc((void**)&d_x, tcbsr.ncols * sizeof(double));
        cudaMalloc((void**)&d_y, tcbsr.nrows * sizeof(double));

        cudaMemcpy(d_x, x, tcbsr.ncols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, tcbsr.nrows * sizeof(double), cudaMemcpyHostToDevice);
        int blockSize = 4*WARP_SIZE;
        int gridSize = (tcbsr.warpCount*32 + blockSize - 1) / blockSize;
        // printf("gridSize %d blockSize %d\n", gridSize, blockSize);
        for (int i = 0; i < warmup; i++) {
            TcSPMVKernel1x2BalCC<int32_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
            cudaDeviceSynchronize();
        }
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repeat; i++) {
            TcSPMVKernel1x2BalCC<int32_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
            cudaDeviceSynchronize();
        }

        auto end = std::chrono::high_resolution_clock::now(); 
        auto elapsed = end - start;
        avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        CHECK(cudaGetLastError());
        cudaMemcpy(y, d_y, tcbsr.nrows * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(warpTargetRow);
        cudaFree(warpBlcPtr);
        cudaFree(blcIdx);
        cudaFree(blcVal);
        cudaFree(d_x);
        cudaFree(d_y);
    }
    return 0;
}

int SpmvDyTileBal(TcBSR<double, int32_t>& tcbsr, double* x, double* y, int tile_m, int tile_k, int warmup, int repeat, double &avgTime) {
    double *blcVal;

#ifdef DEBUG
    printf("nrows:%d ncols:%d\n", tcbsr.nrows, tcbsr.ncols);
    printf(" tcbsr.warpCount  %d\n",  tcbsr.warpCount );
#endif
    // using indexType = int32_t;
    if (tcbsr.ncols < INT16_MAX && tcbsr.nrows < INT16_MAX)  {
        int16_t *blcIdx;
        int32_t *warpBlcPtr;
        int16_t *targetRowByWarpInt16, *blcIdxfp16, *warpTargetRow;
        blcIdxfp16 = (int16_t*)malloc(tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int16_t));
        targetRowByWarpInt16 = (int16_t*)malloc(tcbsr.warpCount * sizeof(int16_t));
        for (int i = 0; i < tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M; i++) {
            blcIdxfp16[i] =  tcbsr.blcIdx[i];
        }
        for (int i = 0; i < tcbsr.warpCount; i++) {
            targetRowByWarpInt16[i] = tcbsr.targetRowByWarp[i];
        }
        cudaMalloc((void**)&warpTargetRow, tcbsr.warpCount * sizeof(int16_t));
        cudaMalloc((void**)&warpBlcPtr, (tcbsr.warpCount + 1) * sizeof(int32_t));
        cudaMalloc((void**)&blcIdx, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int16_t));
        cudaMalloc((void**)&blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double));

        cudaMemcpy(warpTargetRow, targetRowByWarpInt16, tcbsr.warpCount * sizeof(int16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(warpBlcPtr, tcbsr.warpBlcPtr, (tcbsr.warpCount + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcIdx, blcIdxfp16, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcVal, tcbsr.blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double), cudaMemcpyHostToDevice);

        double* d_x, *d_y;
        cudaMalloc((void**)&d_x, tcbsr.ncols * sizeof(double));
        cudaMalloc((void**)&d_y, tcbsr.nrows * sizeof(double));

        cudaMemcpy(d_x, x, tcbsr.ncols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, tcbsr.nrows * sizeof(double), cudaMemcpyHostToDevice);
        int blockSize = 4*WARP_SIZE;
        int gridSize = (tcbsr.warpCount + 3) / 4;
        CHECK(cudaGetLastError());

        if (tile_m == 2) {
            for (int i = 0; i < warmup; i++) {
                TcSPMVKernel2x16Bal<int16_t><<<gridSize, blockSize>>>(warpTargetRow ,warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; i++) {
                TcSPMVKernel2x16Bal<int16_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }

            auto end = std::chrono::high_resolution_clock::now(); 
            auto elapsed = end - start;
            avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        } else if (tile_m == 4) { 
            // printf("tile_m %d\n", tile_m);
            for (int i = 0; i < warmup; i++) {
                TcSPMVKernel1x2BalCC<int16_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; i++) {
                TcSPMVKernel1x2BalCC<int16_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }

            auto end = std::chrono::high_resolution_clock::now(); 
            auto elapsed = end - start;
            avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        } else if (tile_m == 8) {
            for (int i = 0; i < warmup; i++) {
                TcSPMVKernel2x1Balance<int16_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; i++) {
                TcSPMVKernel2x1Balance<int16_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }

            auto end = std::chrono::high_resolution_clock::now(); 
            auto elapsed = end - start;
            avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        } else if (tile_m == 16) {
            for (int i = 0; i < warmup; i++) {
                TcSPMVKernel16x2Bal<int16_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; i++) {
                TcSPMVKernel16x2Bal<int16_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }

            auto end = std::chrono::high_resolution_clock::now(); 
            auto elapsed = end - start;
            avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        }
        CHECK(cudaGetLastError());
        cudaMemcpy(y, d_y, tcbsr.nrows * sizeof(double), cudaMemcpyDeviceToHost);
        // cudaFree(warpTargetRow);
        cudaFree(warpBlcPtr);
        cudaFree(blcIdx);
        cudaFree(blcVal);
        cudaFree(d_x);
        cudaFree(d_y);
    } else {
        int32_t *warpTargetRow, *warpBlcPtr, *blcIdx;
        cudaMalloc((void**)&warpTargetRow, tcbsr.warpCount * sizeof(int32_t));
        cudaMalloc((void**)&warpBlcPtr, (tcbsr.warpCount + 1) * sizeof(int32_t));
        cudaMalloc((void**)&blcIdx, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int32_t));
        cudaMalloc((void**)&blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double));

        cudaMemcpy(warpTargetRow, tcbsr.targetRowByWarp, tcbsr.warpCount * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(warpBlcPtr, tcbsr.warpBlcPtr, (tcbsr.warpCount + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcIdx, tcbsr.blcIdx, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcVal, tcbsr.blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double), cudaMemcpyHostToDevice);
        // for (int i =0; i < tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M ; i++) {
        //     printf("%d\n", tcbsr.blcIdx[i]);
        // }
        double* d_x, *d_y;
        cudaMalloc((void**)&d_x, tcbsr.ncols * sizeof(double));
        cudaMalloc((void**)&d_y, tcbsr.nrows * sizeof(double));

        cudaMemcpy(d_x, x, tcbsr.ncols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, tcbsr.nrows * sizeof(double), cudaMemcpyHostToDevice);
        int blockSize = 4*WARP_SIZE;
        int gridSize = (tcbsr.warpCount + 3) / 4;
        // printf("gridSize %d blockSize %d\n", gridSize, blockSize);
        if (tile_m == 2) {
            for (int i = 0; i < warmup; i++) {
                TcSPMVKernel2x16Bal<int32_t><<<gridSize, blockSize>>>(warpTargetRow ,warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; i++) {
                TcSPMVKernel2x16Bal<int32_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }

            auto end = std::chrono::high_resolution_clock::now(); 
            auto elapsed = end - start;
            avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        } else if (tile_m == 4) { 
            for (int i = 0; i < warmup; i++) {
                TcSPMVKernel1x2BalCC<int32_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; i++) {
                TcSPMVKernel1x2BalCC<int32_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }

            auto end = std::chrono::high_resolution_clock::now(); 
            auto elapsed = end - start;
            avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        } else if (tile_m == 8) {
            for (int i = 0; i < warmup; i++) {
                TcSPMVKernel2x1Balance<int32_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; i++) {
                TcSPMVKernel2x1Balance<int32_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }

            auto end = std::chrono::high_resolution_clock::now(); 
            auto elapsed = end - start;
            avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        } else if (tile_m == 16) {
            for (int i = 0; i < warmup; i++) {
                TcSPMVKernel16x2Bal<int32_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; i++) {
                TcSPMVKernel16x2Bal<int32_t><<<gridSize, blockSize>>>(warpTargetRow, warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }

            auto end = std::chrono::high_resolution_clock::now(); 
            auto elapsed = end - start;
            avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        }

        // auto end = std::chrono::high_resolution_clock::now(); 
        // auto elapsed = end - start;
        // avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        CHECK(cudaGetLastError());
        cudaMemcpy(y, d_y, tcbsr.nrows * sizeof(double), cudaMemcpyDeviceToHost);
        // cudaFree(warpTargetRow);
        cudaFree(warpBlcPtr);
        cudaFree(blcIdx);
        cudaFree(blcVal);
        cudaFree(d_x);
        cudaFree(d_y);
    }
    return 0;
}


int SpmvDyTile(TcBSR<double, int32_t>& tcbsr, double* x, double* y, int tile_m, int tile_k, int warmup, int repeat, double &avgTime) {
    double *blcVal;

#ifdef DEBUG
    printf("nrows:%d ncols:%d\n", tcbsr.nrows, tcbsr.ncols);
    printf(" tcbsr.warpCount  %d\n",  tcbsr.warpCount );
#endif
    // using indexType = int32_t;
    if (tcbsr.ncols < INT16_MAX && tcbsr.nrows < INT16_MAX)  {
        int16_t *blcIdx;
        int32_t *warpBlcPtr;
        int16_t *targetRowByWarpInt16, *blcIdxfp16;
        blcIdxfp16 = (int16_t*)malloc(tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int16_t));
        for (int i = 0; i < tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M; i++) {
            blcIdxfp16[i] =  tcbsr.blcIdx[i];
        }
        // cudaMalloc((void**)&warpTargetRow, tcbsr.warpCount * sizeof(int16_t));
        cudaMalloc((void**)&warpBlcPtr, (tcbsr.brow + 1) * sizeof(int32_t));
        cudaMalloc((void**)&blcIdx, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int16_t));
        cudaMalloc((void**)&blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double));

        // cudaMemcpy(warpTargetRow, targetRowByWarpInt16, tcbsr.warpCount * sizeof(int16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(warpBlcPtr, tcbsr.blcPtr, (tcbsr.brow + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcIdx, blcIdxfp16, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcVal, tcbsr.blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double), cudaMemcpyHostToDevice);

        double* d_x, *d_y;
        cudaMalloc((void**)&d_x, tcbsr.ncols * sizeof(double));
        cudaMalloc((void**)&d_y, tcbsr.nrows * sizeof(double));

        cudaMemcpy(d_x, x, tcbsr.ncols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, tcbsr.nrows * sizeof(double), cudaMemcpyHostToDevice);
        int blockSize = 4*WARP_SIZE;
        int gridSize = (tcbsr.brow*32 + blockSize - 1) / blockSize;
        if (tile_k == 16) {
            for (int i = 0; i < warmup; i++) {
                TcSPMVKernel2x16CC<int16_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; i++) {
                TcSPMVKernel2x16CC<int16_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto end = std::chrono::high_resolution_clock::now(); 
            auto elapsed = end - start;
            avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        } else if (tile_k == 8) {
            for (int i = 0; i < warmup; i++) {
                TcSPMVKernel1x2CC<int16_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; i++) {
                TcSPMVKernel1x2CC<int16_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto end = std::chrono::high_resolution_clock::now(); 
            auto elapsed = end - start;
            avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        } else if (tile_k == 4) {
            for (int i = 0; i < warmup; i++) {
                TcSPMVKernel2x1CC<int16_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; i++) {
                TcSPMVKernel2x1CC<int16_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto end = std::chrono::high_resolution_clock::now(); 
            auto elapsed = end - start;
            avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        } else if (tile_k == 2) { 
            for (int i = 0; i < warmup; i++) {
                TcSPMVKernel16x2CC<int16_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; i++) {
                TcSPMVKernel16x2CC<int16_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto end = std::chrono::high_resolution_clock::now(); 
            auto elapsed = end - start;
            avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        }

        CHECK(cudaGetLastError());
        cudaMemcpy(y, d_y, tcbsr.nrows * sizeof(double), cudaMemcpyDeviceToHost);
        // cudaFree(warpTargetRow);
        cudaFree(warpBlcPtr);
        cudaFree(blcIdx);
        cudaFree(blcVal);
        cudaFree(d_x);
        cudaFree(d_y);
    } else {
        int32_t *warpTargetRow, *warpBlcPtr, *blcIdx;
        // cudaMalloc((void**)&warpTargetRow, tcbsr.warpCount * sizeof(int32_t));
        cudaMalloc((void**)&warpBlcPtr, (tcbsr.brow + 1) * sizeof(int32_t));
        cudaMalloc((void**)&blcIdx, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int32_t));
        cudaMalloc((void**)&blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double));

        // cudaMemcpy(warpTargetRow, tcbsr.targetRowByWarp, tcbsr.warpCount * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(warpBlcPtr, tcbsr.blcPtr, (tcbsr.brow + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcIdx, tcbsr.blcIdx, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcVal, tcbsr.blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double), cudaMemcpyHostToDevice);
        // for (int i =0; i < tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M ; i++) {
        //     printf("%d\n", tcbsr.blcIdx[i]);
        // }
        double* d_x, *d_y;
        cudaMalloc((void**)&d_x, tcbsr.ncols * sizeof(double));
        cudaMalloc((void**)&d_y, tcbsr.nrows * sizeof(double));

        cudaMemcpy(d_x, x, tcbsr.ncols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, tcbsr.nrows * sizeof(double), cudaMemcpyHostToDevice);
        int blockSize = 4*WARP_SIZE;
        int gridSize = (tcbsr.brow*32 + blockSize - 1) / blockSize;
        // printf("gridSize %d blockSize %d\n", gridSize, blockSize);

        if (tile_k == 16) {
            for (int i = 0; i < warmup; i++) {
                TcSPMVKernel2x16CC<int32_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; i++) {
                TcSPMVKernel2x16CC<int32_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto end = std::chrono::high_resolution_clock::now(); 
            auto elapsed = end - start;
            avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        } else if (tile_k == 8) {
            for (int i = 0; i < warmup; i++) {
                TcSPMVKernel1x2CC<int32_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; i++) {
                TcSPMVKernel1x2CC<int32_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto end = std::chrono::high_resolution_clock::now(); 
            auto elapsed = end - start;
            avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        } else if (tile_k == 4) { 
            for (int i = 0; i < warmup; i++) {
                TcSPMVKernel2x1CC<int32_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; i++) {
                TcSPMVKernel2x1CC<int32_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto end = std::chrono::high_resolution_clock::now(); 
            auto elapsed = end - start;
            avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        }
        else if (tile_k == 2) { 
            for (int i = 0; i < warmup; i++) {
                TcSPMVKernel16x2CC<int32_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; i++) {
                TcSPMVKernel16x2CC<int32_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
                cudaDeviceSynchronize();
            }
            auto end = std::chrono::high_resolution_clock::now(); 
            auto elapsed = end - start;
            avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        }
        CHECK(cudaGetLastError());
        cudaMemcpy(y, d_y, tcbsr.nrows * sizeof(double), cudaMemcpyDeviceToHost);
        // cudaFree(warpTargetRow);
        cudaFree(warpBlcPtr);
        cudaFree(blcIdx);
        cudaFree(blcVal);
        cudaFree(d_x);
        cudaFree(d_y);
    }
    return 0;
}



int TcSPMV1x2(TcBSR<double, int32_t>& tcbsr, double* x, double* y, int warmup, int repeat, double &avgTime) {
    double *blcVal;

#ifdef DEBUG
    printf("nrows:%d ncols:%d\n", tcbsr.nrows, tcbsr.ncols);
    printf(" tcbsr.warpCount  %d\n",  tcbsr.warpCount );
#endif
    // using indexType = int32_t;
    if (tcbsr.ncols < INT16_MAX && tcbsr.nrows < INT16_MAX)  {
        int16_t *warpTargetRow, *blcIdx;
        int32_t *warpBlcPtr;

        int16_t *blcIdxfp16;
        blcIdxfp16 = (int16_t*)malloc(tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int16_t));
        for (int i = 0; i < tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M; i++) {
            blcIdxfp16[i] =  tcbsr.blcIdx[i];
        }
        // cudaMalloc((void**)&warpTargetRow, tcbsr.warpCount * sizeof(int16_t));
        cudaMalloc((void**)&warpBlcPtr, (tcbsr.brow + 1) * sizeof(int32_t));
        cudaMalloc((void**)&blcIdx, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int16_t));
        cudaMalloc((void**)&blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double));

        // cudaMemcpy(warpTargetRow, targetRowByWarpInt16, tcbsr.warpCount * sizeof(int16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(warpBlcPtr, tcbsr.blcPtr, (tcbsr.brow + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcIdx, blcIdxfp16, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcVal, tcbsr.blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double), cudaMemcpyHostToDevice);

        double* d_x, *d_y;
        cudaMalloc((void**)&d_x, tcbsr.ncols * sizeof(double));
        cudaMalloc((void**)&d_y, tcbsr.nrows * sizeof(double));

        cudaMemcpy(d_x, x, tcbsr.ncols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, tcbsr.nrows * sizeof(double), cudaMemcpyHostToDevice);
        int blockSize = 4*WARP_SIZE;
        int gridSize = (tcbsr.brow*32 + blockSize - 1) / blockSize;
        for (int i = 0; i < warmup; i++) {
            TcSPMVKernel1x2CC<int16_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
            cudaDeviceSynchronize();
        }
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repeat; i++) {
            TcSPMVKernel1x2CC<int16_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
            cudaDeviceSynchronize();
        }

        auto end = std::chrono::high_resolution_clock::now(); 
        auto elapsed = end - start;
        avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        CHECK(cudaGetLastError());
        cudaMemcpy(y, d_y, tcbsr.nrows * sizeof(double), cudaMemcpyDeviceToHost);
        // cudaFree(warpTargetRow);
        cudaFree(warpBlcPtr);
        cudaFree(blcIdx);
        cudaFree(blcVal);
        cudaFree(d_x);
        cudaFree(d_y);
    } else {
        int32_t *warpBlcPtr, *blcIdx;
        // cudaMalloc((void**)&warpTargetRow, tcbsr.warpCount * sizeof(int32_t));
        cudaMalloc((void**)&warpBlcPtr, (tcbsr.brow + 1) * sizeof(int32_t));
        cudaMalloc((void**)&blcIdx, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int32_t));
        cudaMalloc((void**)&blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double));

        // cudaMemcpy(warpTargetRow, tcbsr.targetRowByWarp, tcbsr.warpCount * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(warpBlcPtr, tcbsr.blcPtr, (tcbsr.brow + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcIdx, tcbsr.blcIdx, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcVal, tcbsr.blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double), cudaMemcpyHostToDevice);
        // for (int i =0; i < tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M ; i++) {
        //     printf("%d\n", tcbsr.blcIdx[i]);
        // }
        double* d_x, *d_y;
        cudaMalloc((void**)&d_x, tcbsr.ncols * sizeof(double));
        cudaMalloc((void**)&d_y, tcbsr.nrows * sizeof(double));

        cudaMemcpy(d_x, x, tcbsr.ncols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, tcbsr.nrows * sizeof(double), cudaMemcpyHostToDevice);
        int blockSize = 4*WARP_SIZE;
        int gridSize = (tcbsr.brow + 4 - 1) / 4;
        // printf("gridSize %d blockSize %d\n", gridSize, blockSize);
        for (int i = 0; i < warmup; i++) {
            TcSPMVKernel1x2CC<int32_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
            cudaDeviceSynchronize();
        }
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repeat; i++) {
            TcSPMVKernel1x2CC<int32_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
            cudaDeviceSynchronize();
        }

        auto end = std::chrono::high_resolution_clock::now(); 
        auto elapsed = end - start;
        avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        CHECK(cudaGetLastError());
        cudaMemcpy(y, d_y, tcbsr.nrows * sizeof(double), cudaMemcpyDeviceToHost);
        // cudaFree(warpTargetRow);
        cudaFree(warpBlcPtr);
        cudaFree(blcIdx);
        cudaFree(blcVal);
        cudaFree(d_x);
        cudaFree(d_y);
    }
    return 0;
}

int TcSPMV1x2Level0(TcBSR<double, int32_t>& tcbsr, double* x, double* y, int warmup, int repeat, double &avgTime) {
    double *blcVal;


    // using indexType = int32_t;
    if (tcbsr.ncols < INT16_MAX)  {
        int16_t *blcIdx;
        int32_t *warpBlcPtr;
#ifdef DEBUG
    printf("nrows:%d ncols:%d\n", tcbsr.nrows, tcbsr.ncols);
    printf(" tcbsr.warpCount  %d\n",  tcbsr.warpCount );
#endif
        int16_t *blcIdxfp16;
        blcIdxfp16 = (int16_t*)malloc(tcbsr.blockCount * sizeof(int16_t));
        for (int i = 0; i < tcbsr.blockCount; i++) {
            blcIdxfp16[i] =  tcbsr.blcIdx[i];
        }
        cudaMalloc((void**)&warpBlcPtr, (tcbsr.warpCount + 1) * sizeof(int32_t));
        cudaMalloc((void**)&blcIdx, tcbsr.blockCount * sizeof(int16_t));
        cudaMalloc((void**)&blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double));

        cudaMemcpy(warpBlcPtr, tcbsr.blcPtr, (tcbsr.warpCount + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcIdx, blcIdxfp16, tcbsr.blockCount * sizeof(int16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcVal, tcbsr.blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double), cudaMemcpyHostToDevice);
        double* d_x, *d_y;
        cudaMalloc((void**)&d_x, tcbsr.ncols * sizeof(double));
        cudaMalloc((void**)&d_y, tcbsr.nrows * sizeof(double));

        // cudaMemcpy(d_x, x, tcbsr.ncols * sizeof(double), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_y, y, tcbsr.nrows * sizeof(double), cudaMemcpyHostToDevice);
        int blockSize = 4*WARP_SIZE;
        int gridSize = (tcbsr.warpCount + 4 - 1) / 4;
        for (int i = 0; i < warmup; i++) {
            TcSPMVKernelLevel01x2CC<int16_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
            cudaDeviceSynchronize();
        }
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repeat; i++) {
            TcSPMVKernelLevel01x2CC<int16_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.brow, tcbsr.nrows, d_x, d_y);
            cudaDeviceSynchronize();
        }

        auto end = std::chrono::high_resolution_clock::now(); 
        auto elapsed = end - start;
        avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        CHECK(cudaGetLastError());
        cudaMemcpy(y, d_y, tcbsr.nrows * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(warpBlcPtr);
        cudaFree(blcIdx);
        cudaFree(blcVal);
        cudaFree(d_x);
        cudaFree(d_y);
    } else {
        int32_t *warpBlcPtr, *blcIdx;
#ifdef DEBUG
        printf("warpCount %d blockCount:%d\n", tcbsr.warpCount, tcbsr.blockCount);
#endif 
        cudaMalloc((void**)&warpBlcPtr, (tcbsr.warpCount + 1) * sizeof(int32_t));
        cudaMalloc((void**)&blcIdx, tcbsr.blockCount * sizeof(int32_t));
        cudaMalloc((void**)&blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double));

        cudaMemcpy(warpBlcPtr, tcbsr.blcPtr, (tcbsr.warpCount + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcIdx, tcbsr.blcIdx, tcbsr.blockCount * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(blcVal, tcbsr.blcVal, tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M * sizeof(double), cudaMemcpyHostToDevice);
        // for (int i =0; i < tcbsr.blockCount * tcbsr.TILE_K * tcbsr.TILE_M ; i++) {
        //     printf("%d\n", tcbsr.blcIdx[i]);
        // }
        double* d_x, *d_y;
        cudaMalloc((void**)&d_x, tcbsr.ncols * sizeof(double));
        cudaMalloc((void**)&d_y, tcbsr.nrows * sizeof(double));

        cudaMemcpy(d_x, x, tcbsr.ncols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, tcbsr.nrows * sizeof(double), cudaMemcpyHostToDevice);
        int blockSize = 4*WARP_SIZE;
        int gridSize = (tcbsr.warpCount*32 + blockSize - 1) / blockSize;
        // printf("gridSize %d blockSize %d\n", gridSize, blockSize);
        for (int i = 0; i < warmup; i++) {
            TcSPMVKernelLevel01x2CC<int32_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
            cudaDeviceSynchronize();
        }
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repeat; i++) {
            TcSPMVKernelLevel01x2CC<int32_t><<<gridSize, blockSize>>>(warpBlcPtr, blcIdx, blcVal, tcbsr.warpCount, tcbsr.nrows, d_x, d_y);
            cudaDeviceSynchronize();
        }

        auto end = std::chrono::high_resolution_clock::now(); 
        auto elapsed = end - start;
        avgTime = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / repeat;
        CHECK(cudaGetLastError());
        cudaMemcpy(y, d_y, tcbsr.nrows * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(warpBlcPtr);
        cudaFree(blcIdx);
        cudaFree(blcVal);
        cudaFree(d_x);
        cudaFree(d_y);
    }
    return 0;
}
