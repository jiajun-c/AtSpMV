#include<iostream>
using namespace std;
int main()
{
    //定义一个cuda的设备属性结构体
    int row =128, avg_nnz_in_row = 2;
    int col = row+avg_nnz_in_row;
    int nnz = row * avg_nnz_in_row;
    printf("%%%MatrixMarket matrix coordinate pattern general\n");
    printf("%d %d %d\n", row, col, nnz);
    for (int i = 0; i < row; i++) {
        // if (i < row/2) {
            // for (int j = 0; j < avg_nnz_in_row; j++) {
            //     printf("%d %d\n", i+1, j+1);
            // }
            for (int j = 0; j < avg_nnz_in_row; j++) {
                printf("%d %d\n", i+1, i+j + 1);
            }
        // }
    }
}