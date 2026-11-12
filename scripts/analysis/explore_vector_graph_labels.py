#!/usr/bin/env python3
"""Explore retraining vector ratios using independently timed Graph candidates.

The existing name-based split and feature table are reused. This is an offline
experiment, not a production model or a blind evaluation of all Plan components.
"""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from train_vector_accuracy import enriched

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/model/selection-model/h100-v2-full-2247'
DATA = ROOT / 'results/model/selection-model/vector-accuracy-v1/dataset.csv'
HISTORICAL = ROOT / 'results/model/hybrid-time-model/plan-tune-2247-v4/status'
EXACT = ROOT / 'results/model/selection-model/h100-absolute-reference-v1/provenance/features-exact.csv'
ADDITIONAL = ROOT / 'results/model/selection-model/h100-absolute-reference-v1/provenance/additional-feature-hashes.json'
OUT = ROOT / 'results/model/selection-model/vector-graph-label-exploration/report.json'
WIDTHS = (1, 2, 4, 8, 16, 32)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    features = collections.defaultdict(dict)
    splits = {}
    with DATA.open(newline='') as f:
        for row in csv.DictReader(f):
            name, width = row['dataset'], int(row['width'])
            split = splits.setdefault(name, row['split'])
            if split != row['split'] or width in features[name]:
                raise ValueError(f'inconsistent input {name}')
            features[name][width] = [float(row[f'x{i}']) for i in range(39)]
    with EXACT.open(newline='') as f:
        exact = {(r['dataset'], int(r['threshold'])): r for r in csv.DictReader(f)}
    for name, values in features.items():
        for width, stored in values.items():
            if not np.allclose(stored, enriched(exact,name,width), rtol=1e-12, atol=1e-12):
                raise ValueError(f'feature mismatch {name} width {width}')
    extra_hashes = json.loads(ADDITIONAL.read_text())
    manifest = json.loads((RUN/'manifest.json').read_text())
    for item in manifest:
        name = item['dataset']
        if name not in features:
            splits[name] = 'not_in_training_table'
            features[name] = {w:enriched(exact,name,w) for w in WIDTHS}
    x, labels, names, frozen_predictions = [], [], [], []
    for name in sorted(features):
        if set(features[name]) != set(WIDTHS):
            raise ValueError(f'missing features {name}')
        state = json.loads((RUN/'status'/f'{name}.json').read_text())
        old_path = HISTORICAL/f'{name}.json'
        historical = json.loads(old_path.read_text()) if old_path.exists() else None
        expected_hash = historical.get('matrix_sha256') if historical else extra_hashes.get(name)
        if state['status'] != 'ok' or (historical and historical['status'] != 'ok'):
            raise ValueError(f'failed run {name}')
        if not expected_hash or state.get('matrix_sha256') != expected_hash:
            raise ValueError(f'matrix changed since feature source {name}')
        groups = collections.defaultdict(list)
        with (RUN/'raw'/f'{name}.csv').open(newline='') as f:
            for row in csv.DictReader(f):
                if row['backend'] == 'atbsr' or row['backend'] in {f'csr_vec{w}' for w in WIDTHS}:
                    v = float(row['us'])
                    if not math.isfinite(v) or v <= 0: raise ValueError(f'invalid time {name}')
                    groups[row['order'], row['backend']].append(v)
        orders = sorted({o for o, _ in groups})
        if len(orders) != 2 or any(len(groups[o, label]) != 9 for o in orders for label in ('atbsr', *(f'csr_vec{w}' for w in WIDTHS))):
            raise ValueError(f'incomplete measurements {name}')
        times = {key: statistics.median(v) for key, v in groups.items()}
        ratios = [math.exp(statistics.mean(math.log(times[o,f'csr_vec{w}']/times[o,'atbsr']) for o in orders)) for w in WIDTHS]
        prediction = collections.defaultdict(list)
        with (RUN/'raw'/f'{name}.log').open(newline='') as f:
            for fields in csv.reader(f):
                if len(fields) >= 12 and fields[:3] == ['MODEL_PREDICTION','auto_vector_model_v2','csr_vector']:
                    prediction[int(fields[9])].append(float(fields[10]))
        if set(prediction) != set(WIDTHS) or any(len(prediction[w]) != 2 for w in WIDTHS):
            raise ValueError(f'incomplete frozen predictions {name}')
        frozen_predictions.append([math.exp(statistics.mean(map(math.log,prediction[w]))) for w in WIDTHS])
        names.append(name)
        x.append([features[name][w] for w in WIDTHS])
        labels.append(ratios)
    x, labels, names, frozen_predictions = map(np.asarray, (x, labels, names, frozen_predictions))
    parts = {s: np.asarray([splits[n] == s for n in names]) for s in ('train', 'dev', 'holdout', 'not_in_training_table')}
    def evaluate_prediction(pred, mask):
        actual = labels[mask]
        regret = actual[np.arange(len(actual)), np.argmin(pred, axis=1)] / actual.min(axis=1)
        return {'cases': len(actual), 'mape': float(np.mean(np.abs(pred / actual - 1))),
                'width_regret_geomean': float(np.exp(np.log(regret).mean())),
                'width_within_5pct': int(np.sum(regret <= 1.05))}
    models = {'frozen_v2': {s:evaluate_prediction(frozen_predictions[mask],mask) for s,mask in parts.items() if s != 'train'}}
    chosen_predictions = {}
    for depth, leaf in ((10,5),(14,3),(14,12),(None,20)):
        model = ExtraTreesRegressor(n_estimators=120, max_depth=depth, min_samples_leaf=leaf,
                                    random_state=20261029, n_jobs=1)
        model.fit(x[parts['train']].reshape(-1,39), np.log(labels[parts['train']].reshape(-1)))
        models[f'depth_{depth}_leaf_{leaf}'] = {
            s:evaluate_prediction(np.exp(model.predict(x[parts[s]].reshape(-1,39))).reshape(-1,6),parts[s])
            for s in ('dev','holdout','not_in_training_table')}
        if (depth, leaf) == (14, 3):
            joblib.dump(model, OUT.parent/'candidate.joblib')
            chosen_predictions = {s:np.exp(model.predict(x[parts[s]].reshape(-1,39))).reshape(-1,6)
                                  for s in ('dev','holdout','not_in_training_table')}
    uncertainty = {}
    for part, pred in chosen_predictions.items():
        mask = parts[part]
        actual = labels[mask]
        old_error = np.abs(frozen_predictions[mask] / actual - 1).mean(axis=1)
        new_error = np.abs(pred / actual - 1).mean(axis=1)
        groups = collections.defaultdict(list)
        for name, delta in zip(names[mask], old_error - new_error):
            family = re.sub(r'[\d_\-]+$', '', name.lower()) or name.lower()
            groups[family].append(delta)
        values = np.asarray([(len(v), sum(v)) for v in groups.values()])
        rng = np.random.default_rng(20261029)
        drawn = values[rng.integers(len(values), size=(5000,len(values)))].sum(axis=1)
        delta = drawn[:,1]/drawn[:,0]
        uncertainty[part] = {'cases':len(actual), 'families':len(groups),
                             'mean_mape_reduction':float(np.mean(old_error-new_error)),
                             'ci95':np.quantile(delta,[.025,.975]).tolist(),
                             'regressed_cases':int(np.sum(new_error > old_error))}
    report = {'method': 'Exploration on 2248-case H100 device 3 run; prior table splits retained. '
                        'Two-order geometric mean of median Graph vector/default ratios. Name-based families are approximate; '
                        'independent allocations, event nodes and previously inspected data limit independence. '
                        'Matrix hashes match historical tuning or the recorded additional input hash, and stored features '
                        'match the feature table. Dev and holdout have been evaluated '
                        'and are no longer blind. No model is exported.',
              'feature_table_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
              'additional_feature_table_sha256': hashlib.sha256(EXACT.read_bytes()).hexdigest(),
              'run_metadata_sha256': hashlib.sha256((RUN/'metadata.json').read_bytes()).hexdigest(),
              'candidate_sha256': hashlib.sha256((OUT.parent/'candidate.joblib').read_bytes()).hexdigest(),
              'split_sizes': {s:int(mask.sum()) for s,mask in parts.items()}, 'models': models,
              'chosen_vs_frozen_bootstrap': uncertainty,
              'bootstrap_caveat': 'Name-based family bootstrap only; does not include measurement noise, '
                                  'exploration or data-set sampling bias.'}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
