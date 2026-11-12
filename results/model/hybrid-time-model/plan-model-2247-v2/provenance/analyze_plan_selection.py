#!/usr/bin/env python3
"""Summarize a benchmark_plan_selection run, retaining failures separately."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    root = a.directory
    manifest = json.loads((root/'manifest.json').read_text())
    selection = json.loads((root/'metadata.json').read_text()).get('selection','auto_tuned')
    rows = []
    counts = collections.Counter()
    selected = collections.Counter()
    for item in manifest:
        name = item['dataset']
        status = root/'status'/f'{name}.json'
        if not status.exists():
            counts['pending'] += 1
            continue
        data = json.loads(status.read_text())
        counts[data['status']] += 1
        if data['status'] != 'ok':
            continue
        values = collections.defaultdict(list)
        configs = {}
        with (root/'raw'/f'{name}.csv').open() as f:
            for record in csv.DictReader(f):
                values[record['backend']].append(float(record['us']))
                configs[record['backend']] = record['config']
        times = {k: statistics.median(v) for k, v in values.items()}
        choice = configs[selection].split('/')[0]
        selected[choice] += 1
        rows.append({'dataset': name, 'nnz': item['nnz'], 'selected': choice, 'selected_config': configs[selection],
                     'auto_us': times[selection], 'default_us': times['atbsr'], 'cusparse_us': times['cusparse'],
                     'speedup_vs_default': times['atbsr']/times[selection],
                     'speedup_vs_cusparse': times['cusparse']/times[selection],
                     'speedup_vs_best_baseline': min(times['atbsr'], times['cusparse'])/times[selection]})
    result = {'manifest_total': len(manifest), 'status_counts': dict(counts), 'selected_backends': dict(selected),
              'selection': selection, 'method': 'Actual Plan selection, independently timed against default AtBSR and cuSPARSE; warm Graph, excludes setup. Randomized manifest prefix is not the complete suite.'}
    if rows:
        for key in ('speedup_vs_default', 'speedup_vs_cusparse', 'speedup_vs_best_baseline'):
            v = [r[key] for r in rows]
            result[key] = {'geomean': math.exp(statistics.mean(math.log(x) for x in v)), 'median': statistics.median(v), 'wins': sum(x > 1 for x in v), 'cases': len(v)}
        with (root/'analysis.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    (root/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
