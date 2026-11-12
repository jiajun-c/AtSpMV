#!/usr/bin/env python3
"""Explore medium-row hybrid prediction from historical H100 tuning logs.

This evaluates an oracle regular baseline and tuning-time samples, not actual
new Plan execution; it does not change the production selector.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from fit_hybrid_selector import vector

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--tuning', type=Path, required=True)
p.add_argument('--features', type=Path, required=True)
p.add_argument('--splits', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
with a.splits.open(newline='') as f:
    splits = {}
    for r in csv.DictReader(f):
        if r['dataset'] in splits and splits[r['dataset']] != r['split']:
            raise ValueError('conflicting split')
        splits[r['dataset']] = r['split']
with a.features.open(newline='') as f:
    features = {(r['dataset'], int(r['threshold'])): r for r in csv.DictReader(f)}
widths = [(tile, parts) for tile in range(6) for parts in (4, 8)]
rows, skipped = [], Counter()
for name, split in sorted(splits.items()):
    f128, f1024 = features[name, 128], features[name, 1024]
    # The full tuner adds these candidates only for rows in [128, 1024).
    if int(f128['long_rows']) == int(f1024['long_rows']):
        skipped['no_medium_rows'] += 1
        continue
    state = json.loads((a.tuning/'status'/(name+'.json')).read_text())
    if state['status'] != 'ok':
        skipped[state['status']] += 1
        continue
    order, plan, active, blocks = None, None, None, {}
    for line in (a.tuning/'raw'/(name+'.log')).read_text().splitlines():
        fields = next(csv.reader([line]))
        if fields[0] == 'BENCH_ORDER': order = fields[1]
        elif fields[0] == 'PLAN_PREP': plan = fields[1]
        elif line.startswith('TUNING_BEGIN'): active = [] if plan == 'auto_tuned' else None
        elif fields[0] == 'TUNING' and active is not None: active.append(fields)
        elif line == 'TUNING_END' and active is not None:
            if order in blocks: raise ValueError(f'{name}: duplicate tuning block')
            blocks[order] = active; active = None
    if len(blocks) != 2:
        raise ValueError(f'{name}: missing tuning blocks')
    labels = {k: [] for k in widths}
    for block in blocks.values():
        regular = [float(r[9]) for r in block if r[1] not in ('hybrid_dsm_atbsr','csr_dsm_vector','mixed_tile')]
        regular_best = min(regular)
        for r in block:
            if r[1] == 'hybrid_dsm_atbsr' and r[5:8] == ['256','4','128']:
                key = int(r[2]), int(r[4])
                if key in labels: labels[key].append(float(r[9])/regular_best)
    if any(len(v) != 2 or any(not math.isfinite(x) or x <= 0 for x in v) for v in labels.values()):
        raise ValueError(f'{name}: missing or invalid medium hybrid samples')
    for tile, parts in widths:
        x = vector(f128, tile) + vector(f1024, tile) + [float(tile), math.log2(parts)]
        ratio = math.exp(statistics.mean(map(math.log, labels[tile, parts])))
        rows.append((name, split, tile, parts, ratio, x))
if not rows:
    raise ValueError('no medium-row examples')
a.output.mkdir(parents=True, exist_ok=True)
with (a.output/'dataset.csv').open('w', newline='') as f:
    w = csv.writer(f); w.writerow(['dataset','split','tile','parts','ratio']+[f'x{i}' for i in range(len(rows[0][-1]))])
    for name, split, tile, parts, ratio, x in rows: w.writerow([name, split, tile, parts, ratio, *x])
train = [r for r in rows if r[1] == 'train']
model = ExtraTreesRegressor(n_estimators=100, max_depth=14, min_samples_leaf=3, random_state=20261029, n_jobs=4)
model.fit(np.array([r[-1] for r in train]), np.log([r[4] for r in train]))
joblib.dump(model, a.output/'exploratory.joblib')
summary = {'scope':'Retrospective initialization timings and oracle regular baseline; no new kernel is measured. The group split is approximate and other models may overlap. Model and 5% gate are exploratory, not blind or production-ready.',
           'dataset_sha256':hashlib.sha256((a.output/'dataset.csv').read_bytes()).hexdigest(),
           'model_sha256':hashlib.sha256((a.output/'exploratory.joblib').read_bytes()).hexdigest(),
           'training_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'features_sha256':hashlib.sha256(a.features.read_bytes()).hexdigest(),
           'splits_sha256':hashlib.sha256(a.splits.read_bytes()).hexdigest(),
           'sklearn_version':sklearn.__version__, 'skipped':dict(skipped), 'metrics':{}}
for split in ('train','dev','holdout'):
    group = [r for r in rows if r[1] == split]
    if not group: continue
    pred = np.exp(model.predict(np.array([r[-1] for r in group])))
    actual = np.array([r[4] for r in group])
    by_case = {}
    for row, estimate in zip(group, pred): by_case.setdefault(row[0], []).append((row[4], estimate))
    chosen, oracle, selected = [], [], 0
    for candidates in by_case.values():
        choice = min(candidates, key=lambda x:x[1])
        selected += int(choice[1] < .95)
        chosen.append(choice[0] if choice[1] < .95 else 1.0)
        best = min(x[0] for x in candidates)
        oracle.append(min(best, 1.0))
    regrets = [x/y for x,y in zip(chosen,oracle)]
    summary['metrics'][split] = {'matrices':len(by_case), 'candidate_mape':float(np.mean(np.abs(pred/actual-1))),
        'selected_hybrid':selected, 'geomean_chosen_over_oracle':math.exp(statistics.mean(map(math.log,regrets))),
        'within_5pct':sum(v<=1.05 for v in regrets), 'over_10pct':sum(v>1.10 for v in regrets),
        'geomean_chosen_vs_regular':math.exp(statistics.mean(map(math.log,chosen))),
        'chosen_faster_5pct':sum(v < .95 for v in chosen), 'chosen_slower_5pct':sum(v > 1.05 for v in chosen)}
(a.output/'report.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
