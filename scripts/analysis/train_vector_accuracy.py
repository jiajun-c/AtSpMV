#!/usr/bin/env python3
"""Explore vector-time prediction with group-disjoint training and frozen tests.

No production code is emitted. Initialization-time labels contain noise and
cross-plan timing bias; the final paired tests are not a fully blind test of
other components of the existing unified model.
"""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from fit_hybrid_selector import vector

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'results/model/hybrid-time-model'
OUTPUT = ROOT / 'results/model/selection-model/vector-accuracy-v1'
TUNING = DATA / 'plan-tune-2247-v4'
WIDTHS = (1, 2, 4, 8, 16, 32)
EVALS = (DATA / 'csr-vector-paired', DATA / 'vector-model-independent-120')
EXCLUDED = EVALS + (ROOT / 'results/model/selection-model/h100-paired40',
                    ROOT / 'results/model/selection-model/h100-refined-new40')


def family(name):
    return re.sub(r'[\d_\-]+$', '', name.lower()) or name.lower()


def split(name):
    value = int(hashlib.sha256(('vector-accuracy-20261029:' + family(name)).encode()).hexdigest()[:8], 16) % 10
    return 'train' if value < 7 else 'dev' if value < 9 else 'holdout'


def geo(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def enriched(features, name, width):
    f = features[name, 1024]
    result = vector(f, 3)
    result.extend(math.log1p(float(f[f'short_tiles_{i}'])) for i in range(6))
    result.extend(math.log1p(float(f[f'short_max_tiles_{i}'])) for i in range(6))
    for threshold in (128, 256, 4096):
        t = features[name, threshold]
        rows, nnz = max(int(t['rows']), 1), max(int(t['nnz']), 1)
        result.extend((int(t['long_rows']) / rows, int(t['long_nnz']) / nnz,
                       math.log1p(float(t['longest_short'])), math.log1p(float(t['long_length_squared']))))
    result.append(math.log2(width))
    return result


def old_features(features, name, width):
    return vector(features[name, 1024], 3) + [math.log2(width)]


def load_tuning(name):
    raw = collections.defaultdict(list)
    with (TUNING / 'raw' / f'{name}.csv').open(newline='') as source:
        for row in csv.DictReader(source):
            if row['backend'] == 'atbsr':
                raw[row['order']].append(float(row['us']))
    orders = list(raw)
    blocks, active, plan = [], None, None
    for line in (TUNING / 'raw' / f'{name}.log').read_text().splitlines():
        if line.startswith('PLAN_PREP,'):
            plan = next(csv.reader([line]))[1]
        elif line.startswith('TUNING_BEGIN'):
            active, block_plan = {}, plan
        elif line.startswith('TUNING,') and active is not None:
            fields = next(csv.reader([line]))
            if fields[1] == 'csr_vector': active[int(fields[8])] = float(fields[9])
        elif line == 'TUNING_END' and active is not None:
            if block_plan == 'auto_tuned': blocks.append(active)
            active = None
    if len(orders) != 2 or len(blocks) != 2 or any(set(block) != set(WIDTHS) for block in blocks):
        raise ValueError(f'{name}: incomplete tuning labels')
    baselines = [statistics.median(raw[order]) for order in orders]
    result = [geo(block[width] / baseline for block, baseline in zip(blocks, baselines)) for width in WIDTHS]
    if not all(math.isfinite(value) and value > 0 for value in result): raise ValueError(f'{name}: invalid ratio')
    return result


def score_width(prediction, actual):
    chosen = np.argmin(prediction, axis=1)
    chosen_time = actual[np.arange(len(actual)), chosen]
    best = np.min(actual, axis=1)
    regret = chosen_time / best
    return {'cases': len(actual), 'ratio_mape': float(np.mean(np.abs(prediction / actual - 1))),
            'log_rmse': float(np.sqrt(np.mean((np.log(prediction) - np.log(actual))**2))),
            'width_regret_geomean': geo(regret), 'width_within_5pct': int(np.sum(regret <= 1.05)),
            'width_over_10pct_slower': int(np.sum(regret > 1.1))}


def fallback_score(name, features, references):
    rows = references[name]
    non = [r for r in rows if int(r['candidate']) < 5]
    if not non: raise ValueError(f'{name}: missing unified predictions')
    best = min(non, key=lambda r: float(r['prediction_ratio']))
    score = float(best['prediction_ratio']) if float(best['prediction_ratio']) < .95 else 1.0
    hybrid = []
    for row in rows:
        i = int(row['candidate'])
        if i < 5: continue
        f = features[name, 4096 if i == 7 else 1024]
        if 0 < int(f['long_rows']) * 10 < int(f['rows']): hybrid.append(row)
    if hybrid:
        best = min(hybrid, key=lambda r: float(r['prediction_ratio']))
        if float(best['prediction_ratio']) < score * .9: score = float(best['prediction_ratio'])
    return score


def paired_evaluate(root, models, features, references):
    index = json.loads((root / 'manifest.json').read_text())
    records = []
    for item in index:
        name = item['dataset']
        state = json.loads((root / 'status' / f'{name}.json').read_text())
        if state['status'] != 'ok': raise ValueError(f'{name}: failed paired evaluation')
        samples = collections.defaultdict(list)
        with (root / 'raw' / f'{name}.csv').open(newline='') as source:
            for r in csv.DictReader(source): samples[r['order'], r['backend']].append(float(r['us']))
        orders = ('baseline_first', 'selection_first')
        required = ('auto_model',) + tuple(f'csr_vec{width}' for width in WIDTHS)
        if any((o, label) not in samples for o in orders for label in required):
            raise ValueError(f'{name}: paired data incomplete')
        times = {k: statistics.median(v) for k, v in samples.items()}
        predictions = {}
        for label, model, use_extra in models:
            x = np.asarray([enriched(features, name, w) if use_extra else old_features(features, name, w) for w in WIDTHS])
            values = np.exp(model.predict(x))
            width = WIDTHS[int(np.argmin(values))]
            pick_vector = min(values) < fallback_score(name, features, references)
            selected = f'csr_vec{width}' if pick_vector else 'auto_model'
            speed = geo(times[o, 'auto_model'] / times[o, selected] for o in orders)
            regret = geo(times[o, selected] / min(times[o, b] for b in required) for o in orders)
            predictions[label] = {'selected': selected, 'speedup': speed, 'regret': regret}
        records.append({'dataset': name, 'predictions': predictions})
    summary = {}
    for label, _, _ in models:
        vals = [r['predictions'][label] for r in records]
        summary[label] = {'cases': len(vals), 'speedup_vs_base': geo(v['speedup'] for v in vals),
                          'regret_vs_vector_and_base_oracle': geo(v['regret'] for v in vals),
                          'within_5pct_oracle': sum(v['regret'] <= 1.05 for v in vals),
                          'over_10pct_slower': sum(v['regret'] > 1.10 for v in vals),
                          'selected_vector': sum(v['selected'] != 'auto_model' for v in vals),
                          'slower_than_base_5pct': sum(v['speedup'] < 1 / 1.05 for v in vals)}
    return summary, records


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    features = {(r['dataset'], int(r['threshold'])): r for r in csv.DictReader((DATA / 'features-exact.csv').open())}
    references = collections.defaultdict(list)
    for row in csv.DictReader((DATA / 'unified-reference.csv').open()): references[row['dataset']].append(row)
    excluded = set()
    for root in EXCLUDED:
        excluded.update(family(r['dataset']) for r in json.loads((root / 'manifest.json').read_text()))
    all_names = [r['dataset'] for r in json.loads((TUNING / 'manifest.json').read_text())]
    names = sorted(name for name in all_names if family(name) not in excluded)
    labels = {}
    for name in names:
        state = json.loads((TUNING / 'status' / f'{name}.json').read_text())
        if state['status'] != 'ok': raise ValueError(f'{name}: tuning run failed')
        labels[name] = load_tuning(name)
    partitions = {key: [n for n in names if split(n) == key] for key in ('train', 'dev', 'holdout')}
    assert not ({family(n) for n in partitions['train']} & {family(n) for n in partitions['holdout']})
    training_file = OUTPUT / 'dataset.csv'
    with training_file.open('w', newline='') as target:
        writer = csv.writer(target)
        writer.writerow(['dataset', 'split', 'width', 'ratio'] + [f'x{i}' for i in range(len(enriched(features, names[0], 1)))])
        for name in names:
            for width, ratio in zip(WIDTHS, labels[name]):
                writer.writerow([name, split(name), width, ratio, *enriched(features, name, width)])
    designs = [('original_features', False, 10, 5), ('extended_shallow', True, 10, 5),
               ('extended_deep', True, 14, 3), ('original_deep', False, 14, 3)]
    models, evaluations = [], {}
    for label, use_extra, depth, leaf in designs:
        train_names = partitions['train']
        x = np.asarray([enriched(features,n,w) if use_extra else old_features(features,n,w) for n in train_names for w in WIDTHS])
        y = np.log(np.asarray([ratio for n in train_names for ratio in labels[n]]))
        model = ExtraTreesRegressor(n_estimators=120, max_depth=depth, min_samples_leaf=leaf, random_state=20261028, n_jobs=1)
        model.fit(x, y)
        models.append((label, model, use_extra))
        evaluations[label] = {}
        for part in ('dev', 'holdout'):
            subset = partitions[part]
            pred = np.exp(model.predict(np.asarray([enriched(features,n,w) if use_extra else old_features(features,n,w) for n in subset for w in WIDTHS]))).reshape(-1, 6)
            actual = np.asarray([labels[n] for n in subset])
            evaluations[label][part] = score_width(pred, actual)
    selected = min(designs, key=lambda d: (evaluations[d[0]]['dev']['width_regret_geomean'], evaluations[d[0]]['dev']['log_rmse']))[0]
    best = next(m for m in models if m[0] == selected)
    # Comparison with the deployed frozen forest, trained on its original data.
    old = list(csv.DictReader((DATA / 'vector-model-from-400-groups.training.csv').open()))
    keys = [f'x{i}' for i in range(15)]
    deployed = ExtraTreesRegressor(n_estimators=120, max_depth=10, min_samples_leaf=5, random_state=20261028, n_jobs=1)
    deployed.fit(np.asarray([[float(row[k]) for k in keys] for row in old]), np.log([float(row['ratio']) for row in old]))
    compared = [('deployed', deployed, False), best]
    paired, records = {}, {}
    for root in EVALS:
        paired[root.name], records[root.name] = paired_evaluate(root, compared, features, references)
    joblib.dump(best[1], OUTPUT / 'selected.joblib')
    report = {'method': 'Excludes approximate families of two paired tests and both prior 40-case samples; hash-partitions remaining tuning logs into train/dev/holdout. Selects architecture on dev width regret, then evaluates frozen holdout and independently timed paired vector candidates. Existing non-vector model and some datasets may have historical overlap. No production update.',
              'excluded_families': len(excluded), 'split_counts': {k: len(v) for k, v in partitions.items()},
              'family_counts': {k: len({family(n) for n in v}) for k, v in partitions.items()},
              'dataset_sha256': hashlib.sha256(training_file.read_bytes()).hexdigest(),
              'selected_on_development': selected, 'tuning_label_evaluation': evaluations,
              'paired_evaluation': paired}
    (OUTPUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (OUTPUT / 'paired-decisions.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
