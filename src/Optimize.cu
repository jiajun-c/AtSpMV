#include "optimize.h"
#include "common.h"

void Level2ToBlock(CSRMatrix<double, int32_t>&csr, TcBSR<double, int32_t>&tcbsr) { 
    // TcBSR<double, int32_t> tcbsr;
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
    // printf("Level2\n");
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
    // return tcbsr;
}
void BalanceDyBSR(TcBSR<double, int32_t>& tcbsr, int WARP_CAPACITY) { 
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
    // density = (double(tcbsr.nnz)/(tcbsr.blockCount*32));
    tcbsr.warpCount  = totalWarp;
    tcbsr.warpBlcPtr = warpBlcPtr;
    tcbsr.targetRowByWarp = warpTargetRow;
}

void Level3DiverseTile(CSRMatrix<double, int32_t>&csr,  TcBSR<double, int32_t>&tcbsr, int tile_m, int tile_k) { 
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
    vector<int>rowPtr(csr.nrows+2, 0);
    vector<double>values(csr.nnz, 0.0);
    vector<int32_t>colIdx(csr.nnz, -1);
    for (int i = 0; i < csr.nrows; i++) {
        int oldIdx = nnzPerRow[i].second;
        // printf("%d %d\n",i,  nnzPerRow[i].first);
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
        // int startRow = i;
        int endRow = min(i + tile_m, csr.nrows);
        int maxLen = nnzPerRow[endRow - 1].first;
        blcPtr[i / tile_m + 1] = blcPtr[i/tile_m] + (maxLen + tile_k - 1) / tile_k;
    }
    blockCount = blcPtr[brow];

    // for (int i = 0; i < brow; i++) {
    //     printf("blcPtr[%d] = %d\n", i, blcPtr[i]);
    // }
    double *blockVal = new double[blockCount * 32];
    int32_t *blockIdx = new int32_t[blockCount * 32];
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
}

void MutliLevelOptimize(CSRMatrix<double, int32_t>& csr,CSRMatrix<double, int32_t>&csrNonZero,TcBSR<double, int32_t>&DyBsr, Optimizer& optim) { 
    // CSRMatrix<double, int32_t> csrNonZero;
    // Level 0 remove zero line 
    int zeroRow = 0, nonZeroRow = 0;
    int *newRowPtr = (int*)malloc((csr.nrows+1) * sizeof(int));
    int *recorder = (int*)malloc(csr.nrows * sizeof(int));
    double *newVal = (double*)malloc(csr.nnz * sizeof(double));
    int *newColIdx = (int*)malloc(csr.nnz * sizeof(int));
    memcpy(newVal, csr.val, csr.nnz * sizeof(double));
    memcpy(newColIdx, csr.col_idx, csr.nnz * sizeof(int));
    newRowPtr[0] = 0;
    for (int i = 0; i < csr.nrows; i++) {
        int len = csr.row_ptr[i + 1] - csr.row_ptr[i];
        if (len == 0) {
            zeroRow++;
            continue;
        }
        recorder[nonZeroRow] = i;
        newRowPtr[nonZeroRow+1] = csr.row_ptr[i+1];
        nonZeroRow++;
    }
#ifdef DEBUG
    printf("zeroRow = %d, nonZeroRow = %d\n", zeroRow, nonZeroRow);
#endif    

    optim.useColIdxInt16 = false;
    if (csr.ncols < INT16_MAX) optim.useColIdxInt16 = true;
    if (csr.nnz < INT16_MAX) optim.useRowPtrInt16 = true;
    csrNonZero.row_ptr = newRowPtr;
    csrNonZero.recorder = recorder;
    csrNonZero.nrows = nonZeroRow;
    csrNonZero.col_idx = csr.col_idx;
    csrNonZero.val = csr.val;
    csrNonZero.nnz = csr.nnz;
    csrNonZero.ncols = csr.ncols;
    double stand = 0;
    double avg_nnz = double(csr.nnz) / nonZeroRow;
    int maxLen = -1, minLen = INT32_MAX;
    // printf("now\n");
    for (int i = 0; i < nonZeroRow; i++) {
        int len = newRowPtr[i + 1] - newRowPtr[i];
        maxLen = max(maxLen, len);
        minLen = min(minLen, len);
        stand += (len - avg_nnz) * (len - avg_nnz);
        // if (csr.row_ptr[i+1] - csr.row_ptr[i] > 8)
        // printf(" ,%d", csr.row_ptr[i+1] - csr.row_ptr[i]);
    }
    int maxAvgRatio = maxLen/avg_nnz;

    stand = sqrt(stand/nonZeroRow);
    optim.useFixLen = false;
    if (minLen == maxLen) {
        optim.fixLen = maxLen;
        optim.useFixLen = true;
    }


    // printf("stand = %lf avg_nnz: %lf\n", stand, avg_nnz);
    if (stand < 12 && avg_nnz < 8 && maxAvgRatio < 4) {
        optim.useTile = false;
        return;
    }

    if (stand > 12 || maxAvgRatio >= 4) {
        optim.useBalance = true;
    } else {
        optim.useBalance = false;
    }
    // return {};
    // Level 2 use tile
    optim.useTile = true;
    int brow = (csrNonZero.nrows + 3)/4;
    int *blcPtr = (int *)malloc((brow+1) * sizeof(int));
    blcPtr[0] = 0;
    vector<int>startIdx;
    for (int i = 0; i < csrNonZero.nrows; i += 4) {
        set<int>s;
        int len = min(4, csrNonZero.nrows - i);
        for (int j = 0; j < len; j++) {
            for (int k = csrNonZero.row_ptr[i+j]; k < csrNonZero.row_ptr[i + j + 1]; k++) {
                s.insert((csrNonZero.col_idx[k])/8);
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
    double density = (double)csrNonZero.nnz / (double)(blockCount * 32);
    // printf("density = %lf block:%d\n", density, blockCount);
    if (density >  0.75) {
        optim.useCompressTile = false;
        Level2ToBlock(csrNonZero, DyBsr);
        return;
    }
    optim.useCompressTile = true;
    // Level 3 diverse shape compress tile 
    int tile_k;
    if (optim.useBalance) {
        tile_k = 4;
    } else {
        if (avg_nnz < 48) {
            tile_k = 4;
        } else if (avg_nnz < 96) {
            tile_k = 8;
        } else {
            tile_k = 16;
        }
    }
    // warp size
    int tile_m = 32/tile_k;
    // printf("tile_m = %d, tile_k = %d\n", tile_m, tile_k);
    Level3DiverseTile(csrNonZero,DyBsr, tile_m, tile_k);
    BalanceDyBSR(DyBsr, 8);
    // return dybsr;
}

CSRMatrix<double, int32_t> recorderCSR(CSRMatrix<double, int32_t>& csr, int TILE_M, int TILE_K) {
    vector<pair<int, int>> nnzPerRow;
    for (int i = 0; i < csr.nrows; i++) {
        nnzPerRow.push_back(make_pair(csr.row_ptr[i + 1] - csr.row_ptr[i], i));
    }
    int32_t* recorder = (int*)malloc(csr.nrows * sizeof(int));

    sort(nnzPerRow.begin(), nnzPerRow.end());
    for (int i = 0; i < csr.nrows; i++) {
        recorder[i] = nnzPerRow[i].second;
    }
    // int brow = (csr.nrows + 7)/8;
    int* rowPtr = (int*)malloc((csr.nrows+1) * sizeof(int)); 
    memset(rowPtr, 0, (csr.nrows+1) * sizeof(int));
    double *values = (double*)malloc(csr.nnz * sizeof(double));
    int *colIdx = (int*)malloc(csr.nnz * sizeof(int));
    for (int i = 0; i < csr.nrows; i++) {
        int oldIdx = nnzPerRow[i].second;
        for (int j = 0; j < nnzPerRow[i].first; j++) {
            values[rowPtr[i] + j] = csr.val[csr.row_ptr[oldIdx] + j];
            colIdx[rowPtr[i] + j] = csr.col_idx[csr.row_ptr[oldIdx] + j];
        }
        rowPtr[i + 1] = rowPtr[i] + nnzPerRow[i].first;
    }
    CSRMatrix<double, int32_t> csrRecorder;
    csrRecorder.nrows = csr.nrows;
    csrRecorder.ncols = csr.ncols;
    csrRecorder.nnz = csr.nnz;
    csrRecorder.row_ptr = rowPtr;
    csrRecorder.col_idx = colIdx;
    csrRecorder.val = values;
    csrRecorder.recorder = recorder;
    return csrRecorder;
}

void splitCSR(CSRMatrix<double, int32_t>& csr, CSRMatrix<double, int32_t>& csrForBlock, CSRMatrix<double, int32_t>& csrRemain, int TILE_M, int TILE_K) {
    int blockRow = 0, remainRow = 0;
    int nnzBlock = 0, nnzRemain = 0;
    for (int i = 0; i < csr.nrows; i += TILE_M) {
        int start = i;
        int end = min(i + TILE_M, csr.nrows);
        int tile_nnz = csr.row_ptr[end] - csr.row_ptr[start];
        int maxLen = csr.row_ptr[end] - csr.row_ptr[end-1];
        if (tile_nnz < maxLen * TILE_M * 0.75 || end - start < 2 || maxLen < 8) {
            remainRow += (end - start);
            nnzRemain += tile_nnz;
        } else {
            blockRow += (end - start);
            nnzBlock += tile_nnz;
        }
    }
#ifdef DEBUG
    printf("nnzBlock %d\n", nnzBlock);
    printf("nnzRemain %d\n", nnzRemain);
    printf("blockRow %d\n", blockRow);
    printf("remainRow %d\n", remainRow);
#endif
    int *block_rowPtr = (int*)malloc((blockRow+1) * sizeof(int));
    memset(block_rowPtr, 0, (blockRow+1) * sizeof(int));
    double *block_val = (double*)malloc(nnzBlock * sizeof(double));
    int *block_colIdx = (int*)malloc(nnzBlock * sizeof(int));

    int *remain_rowPtr = (int*)malloc((remainRow+1) * sizeof(int));
    memset(remain_rowPtr, 0, (remainRow+1) * sizeof(int));
    double *remain_val = (double*)malloc(nnzRemain * sizeof(double));
    int *remain_colIdx = (int*)malloc(nnzRemain * sizeof(int));

    int block_now = 0, remain_now = 0;
    for (int i = 0; i < csr.nrows; i += TILE_M) {
        int start = i;
        int end = min(i + TILE_M, csr.nrows);
        int tile_nnz = csr.row_ptr[end] - csr.row_ptr[start];
        int maxLen = csr.row_ptr[end] - csr.row_ptr[end-1];
#ifdef DEBUG
        // printf("tile_nnz %d maxlen:%d\n", tile_nnz, maxLen);
#endif
        if (tile_nnz < maxLen * TILE_M * 0.75 || end - start < 2 || maxLen < 8) {
            for (int j = 0; j < end - start; j++) {
                for (int z = csr.row_ptr[start + j], y = 0; z < csr.row_ptr[start + j + 1]; z++, y++) {
                    remain_colIdx[remain_rowPtr[remain_now] +  y] = csr.col_idx[z];
                    remain_val[remain_rowPtr[remain_now] +  y] = csr.val[z];
                }
                remain_rowPtr[remain_now + 1] = remain_rowPtr[remain_now] + (csr.row_ptr[start + j + 1] - csr.row_ptr[start + j]);
                remain_now++;
            }
        } else {
            for (int j = 0; j < end - start; j++) {
                for (int z = csr.row_ptr[start + j], y = 0; z < csr.row_ptr[start + j + 1]; z++, y++) {
                    block_colIdx[block_rowPtr[block_now] +  y] = csr.col_idx[z];
                    block_val[block_rowPtr[block_now] +  y] = csr.val[z];
                }
                block_rowPtr[block_now + 1] = block_rowPtr[block_now] + (csr.row_ptr[start + j + 1] - csr.row_ptr[start + j]);
                block_now++;
            }
        }
    }
    csrForBlock.nrows = blockRow;
    csrForBlock.ncols = csr.ncols;
    csrForBlock.nnz = nnzBlock;
    csrForBlock.col_idx = block_colIdx;
    csrForBlock.val = block_val;
    csrForBlock.row_ptr = block_rowPtr;

    csrRemain.nrows = remainRow;
    csrRemain.ncols = csr.ncols;
    csrRemain.nnz = nnzRemain;
    csrRemain.col_idx = remain_colIdx;
    csrRemain.val = remain_val;
    csrRemain.row_ptr = remain_rowPtr;
    // printf("splitCSR de\n");
}