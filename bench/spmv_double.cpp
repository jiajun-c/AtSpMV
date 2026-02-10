#include "common.h"
#include "cusp.h"
#include "csrReader.h"
#include "csrspmv.h"
#include "cusp_bsr.h"
#include "utils.h"
#include "amgT.h"
#include "dasp.h"
#include "atspmv.h"
#include "atspmvprod.h"
using namespace std;

// #define DEBUG
std::vector<std::string> splitString(const std::string& str, char delimiter) {
    std::vector<std::string> parts;
    std::stringstream ss(str); // Create a stringstream from the input string
    std::string segment;

    while (std::getline(ss, segment, delimiter)) { // Read segments separated by the delimiter
        parts.push_back(segment);
    }
    return parts;
}
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
    printf("pass\n");
    return true;
}
int main(int argc, char **argv) {
    char* filepath = argv[1];
    CSRMatrix<double, int32_t>csr = read_mtx(filepath);
    double cuspTime, cuspPreTime, atTime, atPreTime;
    string matrix_name =  splitString(string(filepath), '/').back();


    // spmv_fp64_warpper(csr, hX, tcYverfiy, 0, 1, csrspmv_avg_time);
    // verify_res(0.001, 0.001, tcYverfiy, goldenY, csr.nrows);
    // printf("verf\n");
    int warmup = 0;
    int repeat = 1;
    double *x = (double*)malloc(csr.ncols * sizeof(double));
    for (int i = 0; i < csr.ncols; i++) {
        x[i] = 1;
    }
    double *goldenY = (double*)malloc(csr.nrows * sizeof(double));
    memset(goldenY, 0, csr.nrows * sizeof(double));
    int64_t atBuffSize = 0, cuspBuffSize = 0, amgTBuffSize;

    cusparse_spmvfp64(csr, x, goldenY, repeat, cuspTime, cuspPreTime, cuspBuffSize);


    double *aty = (double*)malloc(csr.nrows * sizeof(double));
    memset(aty, 0, csr.nrows * sizeof(double));
    double atPreTimes[6];
    // for (int i = 0; i < 6;i++) {
    //     Spmv(csr, x, aty, warmup, 10, atTime, atPreTime, i);
    //     printf("%f\n", (double)((long)csr.nnz * 2) / (atTime * 1e3));
    // }
    // printf(",%d, %d, %d, %f\n",csr.nrows, csr.ncols, csr.nnz, csr.stand);
    // printf("%f\n", double(csr.avg_nnz))
    Spmv(csr, x, aty, warmup, repeat, atTime, atPreTime, atBuffSize);
    // return 0;
    double dasp_avg_time, dasp_pre_time;
    int *new_order = (int *)malloc(sizeof(int) * csr.nrows);
    spmv_all("", csr.val, csr.row_ptr, csr.col_idx, x, aty, new_order, csr.nrows, csr.ncols, csr.nnz, 4, 0.75, 256, dasp_avg_time,dasp_pre_time);

    double amgTtime, amgTPretime;
    amgT_spmv_fp64(csr.row_ptr, csr.col_idx, csr.val, x, aty, csr.nrows, csr.ncols, csr.nnz, repeat, amgTtime, amgTPretime, amgTBuffSize);
    std::cout << matrix_name <<  "," << csr.nnz << "," << atBuffSize << "," <<  cuspBuffSize << "," << amgTBuffSize << "\n";
    printf("stand : %f cuspPre:%f atPreTime:%f amgTPretime:%f\n",csr.stand, cuspPreTime, atPreTime,amgTPretime);
    printf("time: cusp:%f,at:%f,amg:%f dasp: %f\n",cuspTime, atTime, amgTtime, dasp_avg_time);
    // printf(",%f, %f, %f, %f,",(double)((long)csr.nnz * 2) / (atTime * 1e3), (double)((long)csr.nnz * 2) / (amgTtime * 1e3), (double)((long)csr.nnz * 2) / (dasp_avg_time * 1e3),(double)((long)csr.nnz * 2) / (cuspTime * 1e3));
    // printf(", %d, %d, %d, %f",csr.nrows, csr.ncols, csr.nnz, csr.stand);
    // printf(" , %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f,", cuspTime,atTime, amgTtime,dasp_avg_time, cuspPreTime, atPreTime, dasp_pre_time, amgTPretime, cuspTime / atTime, cuspPreTime / atPreTime, dasp_pre_time/atPreTime, amgTPretime/atPreTime);
    // for verify
    // memset(aty, 0, csr.nrows * sizeof(double));
    // // AtSpmvProduct(atBsr, x, aty);
    // Spmv(csr, x, aty, 0, 1, atTime, atPreTime);
    // double res = 0,goldenRes=0;
    // for (int i = 0; i < csr.nrows; i++) {
    //     double diff = fabsf(goldenY[i] -  aty[i]);
    //     double tol = 0.001 + double(0.001) * fabsf(goldenY[i]);
    //         // printf("Error: %d %f %f %f\n", i, res[i], ref_res[i], diff);
    //     res += aty[i];
    //     goldenRes += goldenY[i];
    //     if (diff > tol) {
    //         printf("Error: %d %f %f %f\n", i,aty[i],  goldenY[i], diff);
    //         return false; // 任何元素超出容差即返回 false
    //     }
    // }
    // printf("pass\n");
    return 0;
}

// 5 81.920000, 46.811429, 43.690667
// 10 93.622857, 77.101176
// 15 98.304000, 95.906341
// 20 119.156364, 109.226667
// 25 142.469565, 119.156364
// 30 152.409302, 128.501961
// wiki-Talk 
