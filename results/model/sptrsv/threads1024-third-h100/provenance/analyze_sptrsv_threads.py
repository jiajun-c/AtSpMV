#!/usr/bin/env python3
"""Summarize two-order cached tile CTA-size comparisons."""
import argparse
import csv
import io
import json
import math
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('--include-1024', action='store_true')
    a = p.parse_args()
    root = a.directory
    counts = (512, 1024) if a.include_1024 else (128, 256, 512)
    rows = []
    for name in json.loads((root / 'manifest.json').read_text())['names']:
        row = {'name': name, 'status': 'ok'}
        runs = []
        for order in ('forward', 'reverse'):
            path = root / 'cases' / f'{name}-{order}.json'
            if not path.exists():
                row['status'] = 'factor_missing' if (root/'cases'/f'{name}-skip.json').exists() else 'missing'
                break
            data = json.loads(path.read_text())
            if data.get('returncode') != 0:
                row['status'] = 'failed'; break
            measures = {r['variant']: r for r in csv.DictReader(io.StringIO(data.get('stdout', '')))}
            try:
                timings = [float(measures[f'cached-tiles-{n}']['solve_us']) for n in counts]
                if any(not math.isfinite(t) or t <= 0 for t in timings): raise ValueError()
            except (KeyError, ValueError, TypeError):
                row['status'] = 'unsupported_or_invalid'; break
            runs.append(timings)
        if row['status'] == 'ok':
            for i, threads in enumerate(counts):
                row[f'us_{threads}'] = statistics.median(run[i] for run in runs)
            for threads in counts:
                if threads != 512: row[f'speedup_{threads}'] = row['us_512'] / row[f'us_{threads}']
        rows.append(row)
    fields = ['name', 'status'] + [f'us_{n}' for n in counts] + [f'speedup_{n}' for n in counts if n != 512]
    with (root/'analysis.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    valid = [r for r in rows if r['status'] == 'ok']
    summary = {'status_counts': {s: sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})}}
    for threads in (n for n in counts if n != 512):
        speeds = [r[f'speedup_{threads}'] for r in valid]
        summary[f'{threads}_geomean_vs_512'] = math.exp(statistics.mean(math.log(s) for s in speeds)) if speeds else None
        summary[f'{threads}_wins_gt_5pct'] = sum(s > 1.05 for s in speeds)
        summary[f'{threads}_losses_gt_5pct'] = sum(s < 1/1.05 for s in speeds)
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
