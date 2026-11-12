#!/usr/bin/env python3
"""Explain when bounded refinement missed an observed faster Plan."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
p.add_argument('--cutoff', type=float, default=1.10)
a = p.parse_args()
root = a.directory
metadata = json.loads((root/'metadata.json').read_text())
if not (metadata.get('compare_refined_model') and metadata.get('compare_full_tune')):
    raise ValueError('requires refined and full-tune comparisons')
manifest = json.loads((root/'manifest.json').read_text())
status_counts, winning_backends, coverage, records = Counter(), Counter(), Counter(), []
for item in manifest:
    name = item['dataset']
    status_path = root/'status'/(name+'.json')
    if not status_path.exists():
        status_counts['pending'] += 1
        continue
    status = json.loads(status_path.read_text())['status']
    status_counts[status] += 1
    if status != 'ok':
        continue
    groups, configs = defaultdict(list), {}
    with (root/'raw'/(name+'.csv')).open(newline='') as f:
        for r in csv.DictReader(f):
            key = r['order'], r['backend']
            groups[key].append(float(r['us'])); configs[key] = r['config']
    times = {key: statistics.median(values) for key, values in groups.items()}
    orders = sorted({o for o, _ in groups})
    if len(orders) != 2:
        raise ValueError(f'{name}: expected two orders')
    best = {o: min((k for k in times if k[0] == o), key=times.get) for o in orders}
    ratios = [times[o, 'auto_model_refined']/times[best[o]] for o in orders]
    regret = math.exp(statistics.mean(map(math.log, ratios)))
    if regret <= a.cutoff:
        continue
    candidates, order, plan, in_block = defaultdict(set), None, None, False
    for line in (root/'raw'/(name+'.log')).read_text().splitlines():
        fields = next(csv.reader([line]))
        if fields[0] == 'BENCH_ORDER': order = fields[1]
        elif fields[0] == 'PLAN_PREP': plan = fields[1]
        elif line.startswith('TUNING_BEGIN'): in_block = plan == 'auto_model_refined'
        elif fields[0] == 'TUNING' and in_block:
            candidates[order].add(tuple(fields[1:9]) + (fields[10] if fields[1] == 'mixed_tile' else '',))
        elif line == 'TUNING_END': in_block = False
    for o in orders:
        label = best[o][1]
        config = configs[best[o]].split('/')
        if label not in ('auto_tuned', 'auto_tuned_mixed', 'auto_model_refined', 'auto_model', 'auto_vector_model') or len(config) < 8:
            coverage['unresolved_config'] += 1
            continue
        backend = config[0]
        winning_backends[backend] += 1
        parts = tuple(config[:7]) + (config[7].removeprefix('vw='),)
        cap = next((x.removeprefix('mc=') for x in config[8:] if x.startswith('mc=')), '') if backend == 'mixed_tile' else ''
        if backend == 'mixed_tile' and not cap:
            coverage['unresolved_config'] += 1
        else:
            coverage['winner_in_refinement_candidates' if parts+(cap,) in candidates[o] else 'winner_not_in_refinement_candidates'] += 1
    records.append({'dataset':name, 'refined_over_observed_best':regret,
                    'winner_labels':'|'.join(best[o][1] for o in orders),
                    'winner_configs':'|'.join(configs[best[o]] for o in orders)})
summary = {'status_counts':dict(status_counts), 'cases_over_cutoff':len(records), 'cutoff':a.cutoff,
           'winner_backends_per_order':dict(winning_backends), 'candidate_coverage_per_order':dict(coverage),
           'method':'Retrospective comparison of measured Plan Graph times; candidate membership is inferred from refinement tuning logs. Config matching cannot establish causality, and some labels lack resolved configurations.'}
if records:
    with (root/'refined-misses.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=records[0]); w.writeheader(); w.writerows(records)
(root/'refined-misses.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
