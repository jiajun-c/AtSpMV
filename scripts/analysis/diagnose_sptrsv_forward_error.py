#!/usr/bin/env python3
"""Solve a benchmark's rounded FP64 RHS using high-precision substitution.

The input must be real, sorted, lower triangular with explicit nonzero
diagonal. The RHS uses the benchmark's sequential FP64 arithmetic. A high
forward error here indicates the rounded RHS itself is sensitive; it does not
prove that either GPU solver is correct.
"""
import argparse
import json

import mpmath as mp
import numpy as np
from scipy.io import mmread


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('matrix')
    parser.add_argument('--digits', type=int, default=100)
    args = parser.parse_args()
    a = mmread(args.matrix).tocsr()
    if a.shape[0] != a.shape[1]:
        raise ValueError('expected square matrix')
    a.sort_indices()
    n = a.shape[0]
    truth = np.array([.5 + (i * 37 % 101) / 101. for i in range(n)], dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if a.indptr[i] == a.indptr[i + 1] or a.indices[a.indptr[i + 1] - 1] != i:
            raise ValueError('missing diagonal')
        for p in range(a.indptr[i], a.indptr[i + 1]):
            rhs[i] += a.data[p] * truth[a.indices[p]]
    mp.mp.dps = args.digits
    solution = []
    for i in range(n):
        value = mp.mpf(float(rhs[i]))
        for p in range(a.indptr[i], a.indptr[i + 1] - 1):
            value -= mp.mpf(float(a.data[p])) * solution[a.indices[p]]
        solution.append(value / mp.mpf(float(a.data[a.indptr[i + 1] - 1])))
    error = max(abs(solution[i] - mp.mpf(float(truth[i]))) for i in range(n))
    print(json.dumps({'matrix': args.matrix, 'digits': args.digits,
                      'rounded_rhs_high_precision_forward_error': mp.nstr(error, 16)}, indent=2))


if __name__ == '__main__':
    main()
