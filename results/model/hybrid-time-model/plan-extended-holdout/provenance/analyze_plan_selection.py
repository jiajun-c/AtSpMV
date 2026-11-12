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
    metadata = json.loads((root/'metadata.json').read_text())
    selection = metadata.get('selection','auto_tuned')
    extended = metadata.get('extended_baselines',False)
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
                order=record.get('order','baseline_first')
                values[(order,record['backend'])].append(float(record['us']))
                configs[(order,record['backend'])] = record['config']
        orders=sorted({k[0] for k in values})
        times = {k: statistics.median(v) for k, v in values.items()}
        choices = [configs[(order,selection)].split('/')[0] for order in orders]
        choice = choices[0] if len(set(choices))==1 else '|'.join(choices)
        selected[choice] += 1
        geo=lambda seq:math.exp(statistics.mean(math.log(x) for x in seq))
        row={'dataset': name, 'nnz': item['nnz'], 'selected': choice, 'selected_config': '|'.join(configs[(o,selection)] for o in orders),
                     'auto_us': geo(times[(o,selection)] for o in orders), 'default_us': geo(times[(o,'atbsr')] for o in orders), 'cusparse_us': geo(times[(o,'cusparse')] for o in orders),
                     'speedup_vs_default': geo(times[(o,'atbsr')]/times[(o,selection)] for o in orders),
                     'speedup_vs_cusparse': geo(times[(o,'cusparse')]/times[(o,selection)] for o in orders),
                     'speedup_vs_best_baseline': geo(min(times[(o,'atbsr')], times[(o,'cusparse')])/times[(o,selection)] for o in orders)}
        if extended:
            baselines=('atbsr','atbsr_m8','atbsr_m1_balanced','csr_dsm_fixed','csr_two_stage_fixed','cusparse')
            row['speedup_vs_extended_baseline']=geo(min(times[(o,b)] for b in baselines)/times[(o,selection)] for o in orders)
        rows.append(row)
    result = {'manifest_total': len(manifest), 'status_counts': dict(counts), 'selected_backends': dict(selected),
              'selection': selection, 'method': 'Actual Plan selection, independently timed against default AtBSR and cuSPARSE; warm Graph, excludes setup. Ratios across construction orders use the geometric mean. Randomized manifest prefix is not the complete suite.'}
    if rows:
        for key in ('speedup_vs_default', 'speedup_vs_cusparse', 'speedup_vs_best_baseline')+(('speedup_vs_extended_baseline',) if extended else ()):
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
