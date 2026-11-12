#!/usr/bin/env python3
"""Generate exact L factors from SuperLU ILUT, recording all failures."""
import argparse
import json
import time
from pathlib import Path

from scipy.io import mmread, mmwrite
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spilu

parser = argparse.ArgumentParser()
parser.add_argument('--matrix-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--drop-tol', type=float, default=1e-4)
parser.add_argument('--fill-factor', type=float, default=5.)
parser.add_argument('names', nargs='+')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
for name in args.names:
    start = time.monotonic()
    path = args.matrix_root / name / (name + '.mtx')
    info = {'matrix': name, 'source': str(path), 'drop_tol': args.drop_tol,
            'fill_factor': args.fill_factor, 'ordering': 'COLAMD', 'shift': 0}
    try:
        a = csc_matrix(mmread(path), dtype=float)
        a.sum_duplicates()
        info.update(rows=a.shape[0], columns=a.shape[1], original_nnz=a.nnz)
        ilu = spilu(a, drop_tol=args.drop_tol, fill_factor=args.fill_factor, permc_spec='COLAMD')
        lower = ilu.L.tocoo()
        lower.sum_duplicates()
        mmwrite(args.output / (name + '-L.mtx'), lower, field='real', symmetry='general')
        info.update(factor_L_nnz=lower.nnz,
                    note='SuperLU L factor with row/column permutations; not original A')
    except Exception as error:
        info['error'] = repr(error)
    info['elapsed_s'] = time.monotonic() - start
    (args.output / (name + '-factor.json')).write_text(json.dumps(info, indent=2) + '\n')
    print(name, info.get('factor_L_nnz', info.get('error')), flush=True)
