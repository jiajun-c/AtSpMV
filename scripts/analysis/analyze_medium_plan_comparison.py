#!/usr/bin/env python3
"""Summarize paired medium-selector and vector-v2 warm Graph measurements."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', type=Path)
parser.add_argument('--splits', type=Path, required=True)
args = parser.parse_args()
with args.splits.open(newline='') as stream:
    split = {r['dataset']: r['split'] for r in csv.DictReader(stream)}
with (args.run/'index.csv').open(newline='') as stream:
    names = [r['dataset'] for r in csv.DictReader(stream)]
rows, statuses = [], Counter()
for name in names:
    path = args.run/'status'/f'{name}.json'
    state = json.loads(path.read_text())['status'] if path.exists() else 'pending'
    statuses[state] += 1
    if state != 'ok':
        continue
    with (args.run/'raw'/f'{name}.csv').open(newline='') as stream:
        samples = list(csv.DictReader(stream))
    groups = defaultdict(list)
    configs = defaultdict(set)
    for r in samples:
        if r['backend'] in ('auto_vector_model_v2', 'auto_medium_model'):
            groups[r['order'], r['backend']].append(float(r['us']))
            configs[r['backend']].add(r['config'])
    orders = {order for order, _ in groups}
    if len(orders) != 2 or any(not groups[o, k] for o in orders for k in ('auto_vector_model_v2', 'auto_medium_model')):
        raise ValueError(f'{name}: incomplete paired measurements')
    speed = math.exp(statistics.mean(math.log(statistics.median(groups[o, 'auto_vector_model_v2']) /
                                              statistics.median(groups[o, 'auto_medium_model'])) for o in orders))
    rows.append({'dataset': name, 'split': split[name], 'speedup': speed,
                 'baseline_config': '|'.join(sorted(configs['auto_vector_model_v2'])),
                 'medium_config': '|'.join(sorted(configs['auto_medium_model']))})
summary = {'expected': len(names), 'status_counts': dict(statuses), 'groups': {},
           'method': 'Same-run paired warm Graph medians, two construction orders; independent Plan allocations, no setup cost. Train/dev data and previously viewed holdout are not independent blind tests.'}
for group in ('all', 'train', 'dev', 'holdout'):
    subset = [r for r in rows if group == 'all' or r['split'] == group]
    if subset:
        speeds = [r['speedup'] for r in subset]
        summary['groups'][group] = {'cases': len(subset),
            'geomean_speedup_vs_v2': math.exp(statistics.mean(map(math.log, speeds))),
            'faster_5pct': sum(x > 1.05 for x in speeds),
            'slower_5pct': sum(x < 1/1.05 for x in speeds),
            'medium_selected': sum(any((fields := c.split('/'))[0] == 'hybrid_dsm_atbsr' and
                                       len(fields) > 6 and fields[6] == '128'
                                       for c in r['medium_config'].split('|')) for r in subset)}
with (args.run/'comparison.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['dataset', 'split', 'speedup', 'baseline_config', 'medium_config'])
    writer.writeheader()
    writer.writerows(rows)
(args.run/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
