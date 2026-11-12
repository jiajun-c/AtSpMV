#!/usr/bin/env python3
"""Generate U.T from SuperLU ILUT as a non-unit lower solve workload.

Use an existing manifest and its successful factor records to preserve the
original sample and factorization settings. U.T is a new solve operator; it
is not asserted to equal the original input matrix or its lower triangle.
"""
import argparse
import json
import time
from pathlib import Path

from scipy.io import mminfo, mmread, mmwrite
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spilu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--factor-records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in json.loads(args.manifest.read_text())['names']:
        record_path = args.factor_records / (name + '-factor.json')
        record = json.loads(record_path.read_text())
        result = {'matrix': name, 'source_record': str(record_path), 'operator': 'SuperLU U transpose'}
        start = time.monotonic()
        if 'factor_L_nnz' not in record:
            result['error'] = 'original factorization failed: ' + record.get('error', 'unknown')
        else:
            try:
                source = Path(record['source'])
                if mminfo(source)[3] == 'complex':
                    raise ValueError('complex matrix unsupported')
                a = csc_matrix(mmread(source), dtype=float)
                a.sum_duplicates()
                factor = spilu(a, drop_tol=record['drop_tol'], fill_factor=record['fill_factor'],
                               permc_spec=record['ordering'])
                lower = factor.U.transpose().tocsr()
                lower.sum_duplicates()
                lower.eliminate_zeros()
                lower.sort_indices()
                mmwrite(args.output / (name + '-L.mtx'), lower, field='real', symmetry='general')
                result.update(rows=lower.shape[0], nnz=lower.nnz,
                              min_abs_diagonal=float(abs(lower.diagonal()).min()),
                              max_abs_diagonal=float(abs(lower.diagonal()).max()))
            except Exception as error:
                result['error'] = repr(error)
        result['elapsed_s'] = time.monotonic() - start
        (args.output / (name + '-factor.json')).write_text(json.dumps(result, indent=2) + '\n')
        print(name, result.get('nnz', result.get('error')), flush=True)


if __name__ == '__main__':
    main()
