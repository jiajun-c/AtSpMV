#!/usr/bin/env python3
"""Bound exhaustive Auto memory from row-partition tile features.

The selected feature row must contain every matrix row as a short row. This
computes a lower bound for the baseline-first validation process, not a peak
estimate or a guarantee that candidate construction succeeds.
"""
import argparse
import csv
import json
from benchmark_plan_selection import fallback_tuning_minimum_bytes


def bound(row):
    if int(row['long_rows']) != 0 or int(row['short_nnz']) != int(row['nnz']):
        raise ValueError('features must describe every row and nonzero')
    rows, cols, nnz = (int(row[k]) for k in ('rows', 'cols', 'nnz'))
    tiles = [int(row[f'short_tiles_{i}']) for i in range(6)]
    maximum = int(row['longest'])
    if nnz <= 1000000 and rows <= 349651 and cols <= 1785345:
        raise ValueError('matrix is inside the H100 model scope')
    if any(((rows+m-1)//m)*((maximum+32//m-1)//(32//m))*32 > 2147483647
           for m in (1, 2, 4, 8, 16, 32)):
        raise ValueError('cannot prove all tile shapes fit 32-bit offsets')
    index_bytes = 2 if cols < 32767 else 4
    packed_bytes = 8 + index_bytes
    base = fallback_tuning_minimum_bytes(dict(rows=rows, columns=cols, nnz=nnz,
                                              longest_row_nnz=maximum))
    # Two modes for each of six layouts. The three validation baselines use
    # default (at least the smallest layout), fixed m=8 and fixed m=1.
    candidate_padding = 2*packed_bytes*(32*sum(tiles)-6*nnz)
    baseline_padding = packed_bytes*(32*(min(tiles)+tiles[3]+tiles[0])-3*nnz)
    return {'dataset': row['dataset'], 'nnz': nnz, 'tile_counts': tiles,
            'minimum_bytes': base + candidate_padding + baseline_padding,
            'candidate_packed_bytes': 2*32*sum(tiles)*packed_bytes,
            'method': 'Baseline-first validation: input CSR, three packed AtBSR baselines, two CSR baseline Plans, tuning CSR and twelve packed AtBSR tuning candidates. Excludes allocations, padding beyond the stated layouts, temporary buffers, graph state and other candidates.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('features')
    parser.add_argument('--output')
    args = parser.parse_args()
    with open(args.features, newline='') as source:
        features = next(csv.DictReader(source))
    result = json.dumps(bound(features), indent=2) + '\n'
    if args.output:
        with open(args.output, 'w') as output:
            output.write(result)
    else:
        print(result, end='')
