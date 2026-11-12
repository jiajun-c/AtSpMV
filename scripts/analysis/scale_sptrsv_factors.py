#!/usr/bin/env python3
"""Generate non-unit-diagonal test factors by deterministic row scaling.

For each input L, the output is D L with nonzero signed diagonal D.  This
preserves the dependency graph; it is not a new independent matrix sample.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import mmread, mmwrite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--factors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    generated = []
    for name in json.loads(args.manifest.read_text())['names']:
        path = args.factors / (name + '-L.mtx')
        if not path.exists():
            continue
        matrix = mmread(path).tocsr()
        if matrix.shape[0] != matrix.shape[1] or np.iscomplexobj(matrix.data):
            raise ValueError(f'invalid input factor: {path}')
        rows = np.arange(matrix.shape[0])
        # Exact binary scales keep scaling reproducible and avoid additional
        # rounding when computing the transformed matrix.
        scale = np.where(rows % 2, -1., 1.) * np.exp2(rows % 5 - 2)
        matrix.data *= np.repeat(scale, np.diff(matrix.indptr))
        matrix.sort_indices()
        mmwrite(args.output / path.name, matrix, symmetry='general')
        generated.append(name)
    (args.output / 'scaling.json').write_text(json.dumps({
        'description': 'D L, D[i,i] = (-1)^i * 2^((i mod 5)-2)',
        'generated': generated,
    }, indent=2) + '\n')
    print(f'generated {len(generated)} scaled factors')


if __name__ == '__main__':
    main()
