#ifndef CSRREADER_H
#define CSRREADER_H
#include "fast_matrix_market/fast_matrix_market.hpp"
#include "matrixFormat.h"
#include "common.h"

using namespace std;
struct elem {
    int32_t row;
    int32_t col;
    double val;
};

template <typename IT, typename VT>
struct triplet_matrix {
    int32_t nrows = 0, ncols = 0;
    std::vector<IT> rows;
    std::vector<IT> cols;
    std::vector<VT> vals;
};

CSRMatrix<double, int32_t> read_mtx(const std::string& matrix_filename, fast_matrix_market::read_options options = {}) {
    std::ifstream f( matrix_filename);
    options.chunk_size_bytes = 1024*1024;
    triplet_matrix<int32_t, double> triplet;

    fast_matrix_market::read_matrix_market_triplet(f, triplet.nrows, triplet.ncols, triplet.rows, triplet.cols, triplet.vals, options);
    vector<elem> datas;
    int nnz = triplet.vals.size();
    for (int i = 0; i < nnz; i++) {
        datas.push_back(elem{triplet.rows[i], triplet.cols[i], triplet.vals[i]});
    }
    sort(datas.begin(), datas.end(), [](const elem& a, const elem& b) {
        if (a.row == b.row) {
            return a.col < b.col;
        }
        return a.row < b.row;
    });
    for (int i = 0; i < nnz; i++) {
        triplet.rows[i] = datas[i].row;
        triplet.cols[i] = datas[i].col;
        triplet.vals[i] = datas[i].val;
    }
    int32_t* counts;
    counts = new int32_t[triplet.nrows+1];
    memset(counts, 0, (triplet.nrows+1)*sizeof(int32_t));
    for (unsigned int i = 0; i < triplet.vals.size(); i++) {
        counts[triplet.rows[i]+1]++;
    }

    for (int i = 1; i <= triplet.nrows; i++) {
        counts[i] += counts[i-1];
    }
    CSRMatrix<double, int32_t> csr(nnz, triplet.nrows, triplet.ncols);
    csr.nnz = nnz;
    csr.nrows = triplet.nrows;
    csr.ncols = triplet.ncols;
    for (int i = 0;i <= csr.nrows; i++) {
        csr.row_ptr[i] = counts[i];
    }
    for (int i = 0; i < nnz; i++) {
        csr.col_idx[i] = triplet.cols[i];
        csr.val[i] = triplet.vals[i];
    }
    return csr;
}
#endif