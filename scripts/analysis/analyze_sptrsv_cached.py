#!/usr/bin/env python3
"""Summarize all manifest cases without hiding unsupported or invalid cases."""
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
manifest = json.loads((root / 'manifest.json').read_text())
rows = []
for name in manifest['names']:
    row = {'name': name, 'status': '', 'cusparse_graph_us': '', 'tile_graph_us': '',
           'cusparse_host_us': '', 'tile_host_us': '', 'recblock_host_us': '',
           'recblock_valid': False, 'graph_speedup': '', 'host_speedup': '', 'recblock_speedup': ''}
    file = root / 'cases' / (name + '-ours.json')
    if not file.exists():
        row['status'] = 'factorization_failed_or_not_run'; rows.append(row); continue
    obj = json.loads(file.read_text())
    if obj.get('returncode') != 0:
        row['status'] = 'ours_failed_or_timed_out'; rows.append(row); continue
    variants = {v['variant']: v for v in csv.DictReader(io.StringIO(obj['stdout']))}
    if 'cusparse' not in variants or 'cusparse-host' not in variants:
        row['status'] = 'cusparse_missing'; rows.append(row); continue
    row['cusparse_graph_us'] = float(variants['cusparse']['solve_us'])
    row['cusparse_host_us'] = float(variants['cusparse-host']['solve_us'])
    if 'cached-tiles-512' not in variants or 'cached-tiles-host-512' not in variants:
        row['status'] = 'cached_tile_not_emitted'; rows.append(row); continue
    row['status'] = 'ok'
    row['tile_graph_us'] = float(variants['cached-tiles-512']['solve_us'])
    row['tile_host_us'] = float(variants['cached-tiles-host-512']['solve_us'])
    row['graph_speedup'] = row['cusparse_graph_us'] / row['tile_graph_us']
    row['host_speedup'] = row['cusparse_host_us'] / row['tile_host_us']
    recfile = root / 'cases' / (name + '-recblock.json')
    if recfile.exists():
        rec = json.loads(recfile.read_text()); stdout = rec.get('stdout', '')
        m = re.search(r'computation usetime = ([\d.]+) ms', stdout)
        if rec.get('returncode') == 0 and 'x check passed!' in stdout and m:
            row['recblock_valid'] = True
            row['recblock_host_us'] = float(m.group(1)) * 1000
            row['recblock_speedup'] = row['recblock_host_us'] / row['tile_host_us']
    rows.append(row)
with (root / 'analysis.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
def summary(key):
    x = [r[key] for r in rows if isinstance(r[key], float) and r[key] > 0]
    return {'n': len(x), 'wins': sum(v > 1 for v in x),
            'geomean': math.exp(sum(map(math.log, x)) / len(x)) if x else None}
result = {'total': len(rows), 'statuses': {s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})},
          'graph_vs_cusparse': summary('graph_speedup'), 'host_vs_cusparse': summary('host_speedup'),
          'host_vs_modified_recblock': summary('recblock_speedup'),
          'caveat': 'exploratory fixed 512-thread candidate; modified Recblock uses CUDA 11.8, cuSPARSE CUDA 12.4; warm graph vs graph and synchronized host vs host; unsupported excluded'}
(root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
