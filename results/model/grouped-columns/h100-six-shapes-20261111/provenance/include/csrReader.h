#ifndef CSRREADER_H
#define CSRREADER_H

#include "fast_matrix_market/types.hpp"
#include "matrixFormat.h"

#include <string>

CSRMatrix<double, int32_t> read_mtx(
    const std::string& matrix_filename,
    fast_matrix_market::read_options options = {});

#endif
