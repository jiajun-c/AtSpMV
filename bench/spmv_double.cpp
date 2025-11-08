#include "common.h"
#include "cusp.h"
#include "csrReader.h"
#include "tcspmv.h"
#include "csrspmv.h"
#include "dyTileSpmv.h"
#include "optimize.h"
using namespace std;

// #define DEBUG
bool verify_res(double rtol, double atol, double *res, double *ref_res, int nnz) {
    for (int i = 0; i < nnz; i++) {
        double diff = fabsf(res[i] - ref_res[i]);
        double tol = atol + rtol * fabsf(ref_res[i]);
            // printf("Error: %d %f %f %f\n", i, res[i], ref_res[i], diff);

        if (diff > tol) {
            printf("Error: %d %f %f %f\n", i, res[i], ref_res[i], diff);
            return false; // 任何元素超出容差即返回 false
        }
    }
    return true;
}
std::vector<std::string> splitString(const std::string& str, char delimiter) {
    std::vector<std::string> parts;
    std::stringstream ss(str); // Create a stringstream from the input string
    std::string segment;

    while (std::getline(ss, segment, delimiter)) { // Read segments separated by the delimiter
        parts.push_back(segment);
    }
    return parts;
}
int main(int argc, char **argv) {
    Optimizer optim;
    char* filepath = argv[1];
    CSRMatrix<double, int32_t>csr = read_mtx(filepath);
    TcBSR<double, int32_t> optimtcBsr;
    CSRMatrix<double, int32_t>csrNonZero;
    MutliLevelOptimize(csr, csrNonZero, optimtcBsr, optim);
    int repeat = 10;
    string matrix_name =  splitString(string(filepath), '/').back();
    auto nonzeroCSR = removeZeroLine(csr);
    double *hX = (double*)malloc(csr.ncols * sizeof(double));
    for (int i = 0; i < csr.ncols; i++) {
        hX[i] = 1.0;
    }
    int maxLen = -1;
    int minLen = INT32_MAX;
    int zerolen = 0;
    for (int i = 0; i < csr.nrows; i++) {
        int len = csr.row_ptr[i+1] - csr.row_ptr[i];
        maxLen = max(maxLen, len);
        minLen = min(minLen, len);
        if (len == 0) {
            zerolen++;
        }
        // if (csr.row_ptr[i+1] - csr.row_ptr[i] > 8)
        // printf(" ,%d", csr.row_ptr[i+1] - csr.row_ptr[i]);
    }
    int nonzeroRow = csr.nrows - zerolen;
    std::cout << matrix_name;
    // printf("max len: %d, min len: %d\n", maxLen, minLen);
    // printf(" ,%d, %d, %d, %d\n", csr.nrows, csr.ncols, maxLen, minLen);
    // return 0;
    if (maxLen - minLen == 0) {
        csr.useFixLen = true;
        csr.FixLen = maxLen;
    } else {
        csr.useFixLen = false;
        // return 0;
    }
#ifdef DEBUG
    printf("\nzero len: %d\n", zerolen);
#endif
    double avg_len =  double(csr.nnz)/nonzeroRow;
    double stand = 0.0;
    for (int i = 0; i < csr.nrows; i++) {
        int len = csr.row_ptr[i+1] - csr.row_ptr[i];
        if (len == 0) continue;
        // double len = csr.row_ptr[i+1] - csr.row_ptr[i];
        stand += (len - avg_len)*(len - avg_len);
        // if (len > 10000) {
        //     printf("Error: %d %f\n", i, stand);
        // }
    }
    stand = sqrt(stand/nonzeroRow);
    // return 0;
    double density0;
    auto bsr = Level0ToBlock(csr, density0);
    CSRMatrix<double, int32_t> reCSR = recorderCSR(csr, 8, 4);
    CSRMatrix<double, int32_t>csrForBlock, csrRemain;
    // splitCSR(reCSR, csrForBlock, csrRemain,  8, 4);
    // return 0;
    // auto tcbsr = Level1Denser(csr, 8 , 4);
    auto tcbsr = Level1Denser(csr, 8, 4);
    auto tcbsr1x2 = Level1Denser(csr, 4, 8);
    auto tcbsr2x16 = Level1Denser(csr, 2, 16);
    auto tcbsr16x2 = Level1Denser(csr, 16, 2);

    double density2x1, density1x2;
    BalanceTcBSR(tcbsr, 8, density2x1);
    BalanceTcBSR(tcbsr1x2, 8, density1x2);
    BalanceTcBSR(tcbsr2x16, 8, density1x2);
    BalanceTcBSR(tcbsr16x2, 8, density1x2);

    double *goldenY = (double*)malloc(csr.nrows * sizeof(double));
    memset(goldenY, 0.0, csr.nrows * sizeof(double));
    double cusparse_avg_time, csrspmv_bal_avg_time, csrspmv_avg_time;
    double tc2x1AvgTimeBalance, tcAvg1x2TimeBalance, tcAvg2x16TimeBalance, tc2x1AvgTime, tc2x1Level0Time, tcAvgTime1x2, tc2x16AvgTime, tc16x2AvgTime;
    double *tcY = (double*)malloc(csr.nrows * sizeof(double));
    memset(tcY, 0.0, csr.nrows * sizeof(double));
    int warmup = 10;
    cusparse_spmvfp64(csr, hX, goldenY, repeat, cusparse_avg_time);
    double dyTime;
    dyTileSpmv(nonzeroCSR, optimtcBsr, optim, hX, tcY, warmup, repeat, dyTime);

    TcSPMV1x2Level0(bsr, hX, tcY, warmup, repeat, tc2x1Level0Time);

    SpmvDyTile(tcbsr2x16, hX, tcY, 2, 16, warmup, repeat, tc2x16AvgTime);

    SpmvDyTile(tcbsr16x2, hX, tcY, 16, 2, warmup, repeat, tc16x2AvgTime);

    SpmvDyTile(tcbsr1x2, hX, tcY, 4, 8, warmup, repeat,  tcAvgTime1x2);

    SpmvDyTile(tcbsr, hX, tcY, 8, 4, warmup, repeat, tc2x1AvgTime);


    SpmvDyTileBal(tcbsr2x16, hX, tcY, 2, 16, warmup, repeat, tcAvg2x16TimeBalance);

    SpmvDyTileBal(tcbsr, hX, tcY, 8, 4, warmup, repeat, tc2x1AvgTimeBalance);


    SpmvDyTileBal(tcbsr1x2, hX, tcY,4, 8, warmup, repeat,  tcAvg1x2TimeBalance);


    double *csrSpmvY = (double*)malloc(csr.nrows * sizeof(double));
    memset(csrSpmvY, 0.0, csr.nrows * sizeof(double));
    spmv_fp64_balance_warpper(csr, hX, csrSpmvY, warmup, repeat, csrspmv_bal_avg_time);
    reCSR.useFixLen = false;
    spmv_fp64_warpper(csr, hX, csrSpmvY, warmup, repeat, csrspmv_avg_time);
    printf(" ,%d ,%d, %.2f, %d, %d, %d, %f", csr.nrows, csr.ncols, avg_len, minLen, maxLen, zerolen, stand);
    printf(" ,%.2f, %.2f, %.2f",density0, density2x1, density1x2);
    printf(" ,%.2f ,%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n", cusparse_avg_time/tc2x1AvgTimeBalance,  cusparse_avg_time/tcAvg1x2TimeBalance, cusparse_avg_time/tcAvg2x16TimeBalance, cusparse_avg_time/csrspmv_bal_avg_time,cusparse_avg_time/tcAvgTime1x2,  cusparse_avg_time/tc2x1AvgTime, cusparse_avg_time/tc2x16AvgTime, cusparse_avg_time/tc16x2AvgTime, cusparse_avg_time/tc2x1Level0Time, cusparse_avg_time/csrspmv_avg_time, cusparse_avg_time/dyTime);
#ifdef DEBUG
    printf("tcspmv: %.3f us, cusparse: %.3f us csrspmv_bal_avg_time: %.3f\n", tc2x1AvgTimeBalance, cusparse_avg_time, csrspmv_bal_avg_time);
#endif
#define VERIFY
#ifdef VERIFY
    double *tcYverfiy = (double*)malloc(csr.nrows * sizeof(double));
    memset(tcYverfiy, 0.0, csr.nrows * sizeof(double));
    // TcSPMV2x1Balance(tcbsr, hX, tcYverfiy, 0, 1, tc2x1AvgTimeBalance);

    dyTileSpmv(nonzeroCSR, optimtcBsr, optim, hX, tcYverfiy, 0, 1, dyTime);
    // spmv_fp64_warpper(csr, hX, tcYverfiy, 0, 1, csrspmv_avg_time);
    verify_res(0.001, 0.001, tcYverfiy, goldenY, csr.nrows);
    // printf("verf\n");
    // for (int i = 0; i < tcbsr.nrows; i++) {
    //     if (reCSR.recorder[i] < 0 || reCSR.recorder[i] >= csr.nrows) {
    //         printf("Error: %d %d\n", i, reCSR.recorder[i]);
    //     }
    //     double diff = fabsf(tcYverfiy[i] -  goldenY[reCSR.recorder[i]]);
    //     double tol = 0.001 + double(0.001) * fabsf(tcYverfiy[i]);
    //         // printf("Error: %d %f %f %f\n", i, res[i], ref_res[i], diff);

    //     if (diff > tol) {
    //         printf("Error: %d %f %f %f\n", i,tcYverfiy[i],  goldenY[reCSR.recorder[i]], diff);
    //         // return false; // 任何元素超出容差即返回 false
    //     }
    // }
    // printf("pass\n");
#endif
    return 0;
}