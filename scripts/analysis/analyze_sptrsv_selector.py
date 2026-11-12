#!/usr/bin/env python3
"""Audit the actually executed selector, retaining failures and fallbacks."""
import argparse
import csv
import io
import json
import math
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
a = p.parse_args()
root = a.directory
names = json.loads((root / 'manifest.json').read_text())['names']
rows = []
for name in names:
    row = {'name': name, 'status': '', 'backend': '', 'rows': '', 'nnz': '', 'levels': '',
           'cusparse_us': '', 'selected_us': '', 'tile_us': '', 'speedup': '', 'oracle_us': '', 'regret': '',
           'recblock_status': 'not_compared', 'recblock_host_us': '', 'matched_backend_host_us': '', 'recblock_speedup': ''}
    file = root / 'cases' / (name + '-ours.json')
    if not file.exists():
        factor_info = root / 'factors' / (name + '-factor.json')
        factor_error = factor_info.exists() and 'error' in json.loads(factor_info.read_text())
        row['status'] = 'factorization_failed' if factor_error else 'benchmark_pending_or_missing'
        rows.append(row); continue
    data = json.loads(file.read_text())
    if data.get('returncode') != 0:
        row['status'] = 'benchmark_failed_or_timeout'; rows.append(row); continue
    entries = list(csv.DictReader(io.StringIO(data.get('stdout', ''))))
    variants = {r['variant']: r for r in entries}
    selected = [r for r in entries if r['variant'] in ('selector-cached', 'selector-cusparse')]
    if 'cusparse' not in variants or len(selected) != 1:
        row['status'] = 'selector_or_reference_missing'; rows.append(row); continue
    ref, result = variants['cusparse'], selected[0]
    row.update(status='ok', backend=result['variant'], rows=int(ref['n']), nnz=int(ref['nnz']),
               levels=int(ref['levels']), cusparse_us=float(ref['solve_us']), selected_us=float(result['solve_us']))
    row['speedup'] = row['cusparse_us'] / row['selected_us']
    candidates = [row['cusparse_us']]
    if 'cached-tiles-512' in variants:
        row['tile_us'] = float(variants['cached-tiles-512']['solve_us']); candidates.append(row['tile_us'])
    row['oracle_us'] = min(candidates)
    row['regret'] = row['selected_us'] / row['oracle_us'] - 1
    recfile = root / 'cases' / (name + '-recblock.json')
    if recfile.exists():
        rec = json.loads(recfile.read_text()); text = rec.get('stdout', '')
        match = re.search(r'computation usetime = ([\d.]+) ms', text)
        key = 'cached-tiles-host-512' if row['backend'] == 'selector-cached' else 'cusparse-host'
        if rec.get('returncode') == 0 and 'x check passed!' in text and match and key in variants:
            row['recblock_status'] = 'valid'
            row['recblock_host_us'] = float(match.group(1)) * 1000
            row['matched_backend_host_us'] = float(variants[key]['solve_us'])
            row['recblock_speedup'] = row['recblock_host_us'] / row['matched_backend_host_us']
        else:
            row['recblock_status'] = 'incorrect_or_failed'
    rows.append(row)
with (root / 'selector-analysis.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
ok = [r for r in rows if r['status'] == 'ok']
selected_tile = [r for r in ok if r['backend'] == 'selector-cached']
summary = {'manifest_count': len(rows), 'status_counts': {s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})},
           'selected_cached': len(selected_tile), 'selected_cusparse': len(ok) - len(selected_tile),
           'geomean_vs_cusparse': math.exp(sum(math.log(r['speedup']) for r in ok) / len(ok)) if ok else None,
           'regret_gt_5pct': sum(r['regret'] > .05 for r in ok),
           'cached_slower_than_cusparse_5pct': sum(r['selected_us'] > 1.05 * r['cusparse_us'] for r in selected_tile),
           'recblock_valid': sum(r['recblock_status'] == 'valid' for r in ok),
           'recblock_incorrect_or_failed': sum(r['recblock_status'] == 'incorrect_or_failed' for r in ok),
           'recblock_geomean': (math.exp(sum(math.log(r['recblock_speedup']) for r in ok if isinstance(r['recblock_speedup'], float) and r['recblock_speedup'] > 0) / sum(isinstance(r['recblock_speedup'], float) and r['recblock_speedup'] > 0 for r in ok)) if any(isinstance(r['recblock_speedup'], float) and r['recblock_speedup'] > 0 for r in ok) else None),
           'caveat': 'one sequential pass conditional on successful ILU factorization; excludes setup/transfers; selector Graph vs cuSPARSE Graph; Recblock CUDA 11.8 synchronized host vs matched-backend CUDA 12.4 host, different timing and numerical checks'}
(root / 'selector-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
