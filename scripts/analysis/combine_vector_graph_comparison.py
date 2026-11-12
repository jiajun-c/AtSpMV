#!/usr/bin/env python3
"""Combine separately measured Graph-model evaluation and train/dev runs."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path('results/model/selection-model'))
parser.add_argument('--expected',type=int,default=2248)
args = parser.parse_args()
root = args.root
groups = {'evaluation': root/'h100-vector-graph-663',
          'train_dev': root/'h100-vector-graph-train-dev'}
rows, seen, statuses = [], set(), {}
for label, path in groups.items():
    summary = json.loads((path/'graph-comparison.json').read_text())
    statuses[label] = summary['status_counts']
    file = path/'graph-comparison.csv'
    if not file.exists():
        if summary['cases'] != 0: raise ValueError(f'missing comparison rows: {path}')
        continue
    with file.open(newline='') as f:
        source = list(csv.DictReader(f))
    if len(source) != summary['cases']: raise ValueError(f'row count mismatch: {path}')
    for row in source:
        if row['dataset'] in seen: raise ValueError('duplicate matrix across runs')
        seen.add(row['dataset'])
        row['evaluation_group'] = label
        rows.append(row)
if not rows: raise ValueError('no completed cases')
out = root/'h100-vector-graph-all'
out.mkdir(parents=True,exist_ok=True)
with (out/'comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def geo(field): return math.exp(statistics.mean(math.log(float(r[field])) for r in rows))
result = {'expected':args.expected,'cases':len(rows),'status_counts':statuses,
          'method':'Separate H100 runs: evaluation matrices and train/dev matrices. '
                   'Warm Graph, independently allocated candidates and order effects; setup excluded. '
                   'Failed and missing cases excluded. Evaluation data were previously inspected.',
          'graph_speedup_vs_v2':geo('graph_speedup_vs_v2'),
          'graph_speedup_vs_cusparse':geo('graph_speedup_vs_cusparse'),
          'v2_width_ratio_mape':statistics.mean(float(r['v2_width_ratio_mape']) for r in rows),
          'graph_width_ratio_mape':statistics.mean(float(r['graph_width_ratio_mape']) for r in rows)}
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
