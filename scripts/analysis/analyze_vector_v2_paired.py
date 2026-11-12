#!/usr/bin/env python3
"""Analyze actual paired v1/v2 Plan executions by training split."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import statistics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
p.add_argument('--training-dataset', type=Path, required=True)
a = p.parse_args()
manifest = json.loads((a.run/'manifest.json').read_text())
metadata = json.loads((a.run/'metadata.json').read_text())
if not metadata.get('compare_vector_model_v2'):
    raise ValueError('run did not measure v2')
with (a.run/'configs.csv').open(newline='') as f:
    labels = {r[0] for r in csv.reader(f) if r and not r[0].startswith('#')} | {'cusparse'}
splits = {}
with a.training_dataset.open(newline='') as f:
    for r in csv.DictReader(f):
        old = splits.setdefault(r['dataset'], r['split'])
        if old != r['split']:
            raise ValueError(f'inconsistent training split: {r["dataset"]}')
def geo(values):
    return math.exp(statistics.mean(math.log(v) for v in values))
rows, statuses = [], Counter()
for item in manifest:
    name = item['dataset']
    status = a.run/'status'/(name+'.json')
    if not status.exists():
        statuses['pending'] += 1
        continue
    state = json.loads(status.read_text())['status']
    statuses[state] += 1
    if state != 'ok':
        continue
    groups, config = defaultdict(list), {}
    with (a.run/'raw'/(name+'.csv')).open(newline='') as f:
        for r in csv.DictReader(f):
            key = r['order'], r['backend']
            value = float(r['us'])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name}: invalid measurement')
            groups[key].append(value)
            config[key] = r['config']
    orders = sorted({o for o, _ in groups})
    expected = {(order, label) for order in orders for label in labels}
    if len(orders) != 2 or set(groups) != expected or any(len(values) != metadata['trials'] for values in groups.values()):
        raise ValueError(f'{name}: incomplete samples')
    times = {key: statistics.median(v) for key, v in groups.items()}
    v1, v2 = 'auto_vector_model', 'auto_vector_model_v2'
    prep, predictions, references, order = {}, {}, {}, None
    for line in (a.run/'raw'/(name+'.log')).read_text().splitlines():
        fields = next(csv.reader([line]))
        if not fields: continue
        if fields[0] == 'BENCH_ORDER': order = fields[1]
        elif fields[0] == 'PLAN_PREP' and fields[1] in (v1, v2):
            key = order, fields[1]
            if order is None or key in prep: raise ValueError(f'{name}: invalid setup record')
            prep[key] = float(fields[2])
        elif fields[0] == 'MODEL_REFERENCE' and fields[1] in (v1,v2):
            key = order, fields[1]
            if order is None or key in references: raise ValueError(f'{name}: invalid reference record')
            references[key] = (float(fields[2]), int(fields[3]))
        elif fields[0] == 'MODEL_PREDICTION' and fields[1] in (v1,v2):
            config_string = '/'.join(fields[2:9]+['vw='+fields[9]])
            key = order, fields[1], config_string
            if order is None or key in predictions: raise ValueError(f'{name}: duplicate prediction')
            predictions[key] = (float(fields[10]), float(fields[11]))
    if any((o, label) not in prep or not math.isfinite(prep[o,label]) or prep[o,label] <= 0
           for o in orders for label in (v1,v2)):
        raise ValueError(f'{name}: incomplete setup records')
    speedups, regret1, regret2, same = [], [], [], []
    for o in orders:
        best = min(t for (order, _), t in times.items() if order == o)
        speedups.append(times[o, v1] / times[o, v2])
        regret1.append(times[o, v1] / best)
        regret2.append(times[o, v2] / best)
        same.append(config[o, v1] == config[o, v2])
    nnz = int(item['nnz'])
    row = {'dataset': name, 'nnz':nnz,
                 'nnz_bucket':'<10k' if nnz<10000 else '10k-100k' if nnz<100000 else '100k-1M' if nnz<=1000000 else '>1M',
                 'v2_backend':'|'.join(sorted({config[o,v2].split('/')[0] for o in orders})),
                 'training_split': splits.get(name, 'not_in_training_table'),
                 'v2_speedup_vs_v1': geo(speedups), 'v1_over_observed_best': geo(regret1),
                 'v2_over_observed_best': geo(regret2), 'same_config_both_orders': all(same),
                 'v2_setup_over_v1': geo(prep[o,v2] / prep[o,v1] for o in orders),
                 'v1_config': '|'.join(config[o, v1] for o in orders),
                 'v2_config': '|'.join(config[o, v2] for o in orders)}
    if metadata.get('log_model_predictions'):
        if any((o,v2) not in references for o in orders): raise ValueError(f'{name}: missing model reference')
        row['v2_reference_calibrated'] = '|'.join(str(v) for v in sorted({references[o,v2][1] for o in orders}))
        for label, prefix in ((v1,'v1'),(v2,'v2')):
            errors_ratio, errors_absolute = [], []
            for o in orders:
                if (o,label) not in references: raise ValueError(f'{name}: missing model reference')
                value = predictions.get((o,label,config[o,label]))
                if value is None: continue  # Usually the default-AtBSR fallback.
                predicted_ratio, predicted_us = value
                if not all(math.isfinite(x) and x > 0 for x in value): raise ValueError(f'{name}: invalid prediction')
                actual = times[o,label]
                errors_ratio.append(abs(predicted_ratio/(actual/times[o,'atbsr']) - 1))
                errors_absolute.append(abs(predicted_us/actual - 1))
            row[prefix+'_prediction_ratio_mape'] = statistics.mean(errors_ratio) if len(errors_ratio)==2 else ''
            row[prefix+'_prediction_absolute_mape'] = statistics.mean(errors_absolute) if len(errors_absolute)==2 else ''
            if metadata.get('vector_candidates'):
                errors_ratio, errors_absolute, width_regret = [], [], []
                for o in orders:
                    options = []
                    for width in (1,2,4,8,16,32):
                        cfg = f'csr_vector/-1/-1/8/128/1/1024/vw={width}'
                        predicted_ratio, predicted_us = predictions[o,label,cfg]
                        actual = times[o,f'csr_vec{width}']
                        if not all(math.isfinite(x) and x > 0 for x in (predicted_ratio,predicted_us,actual)):
                            raise ValueError(f'{name}: invalid vector prediction or timing')
                        errors_ratio.append(abs(predicted_ratio/(actual/times[o,'atbsr'])-1))
                        errors_absolute.append(abs(predicted_us/actual-1))
                        options.append((predicted_ratio,actual))
                    width_regret.append(min(options,key=lambda x:x[0])[1]/min(x[1] for x in options))
                row[prefix+'_all_width_ratio_mape'] = statistics.mean(errors_ratio)
                row[prefix+'_all_width_absolute_mape'] = statistics.mean(errors_absolute)
                row[prefix+'_width_over_measured_best'] = geo(width_regret)
    for n in (1,100,1000,10000,100000,1000000):
        row[f'v2_total_speedup_n{n}'] = geo((prep[o,v1]+n*times[o,v1])/(prep[o,v2]+n*times[o,v2]) for o in orders)
    rows.append(row)
summary = {'status_counts': dict(statuses), 'total': len(manifest),
           'method': 'Actual v1 and v2 Plans measured within each run and construction order; per-order medians combined geometrically. Oracle is the fastest observed Plan in that order, not all possible kernels. Prediction MAPE covers cases where the chosen configuration matches a logged candidate in both orders; default fallbacks without a prediction are omitted. Actual ratios use independently timed default AtBSR, while absolute estimates use the model reference. Graph events add overhead. Training splits describe the new vector model only; other components and prior inspections may overlap. Total cost estimates use Plan setup plus N warm Graph calls and omit input loading, caller allocations and Graph creation; independent Plan allocation and construction order can affect setup.'}
if metadata.get('vector_candidates'):
    summary['method'] += ' All-width metrics independently time all six vector Plans in both orders; allocations differ from model-selected Plans and affect comparisons.'
for label in ('all', 'train', 'dev', 'holdout', 'not_in_training_table'):
    group = rows if label == 'all' else [r for r in rows if r['training_split'] == label]
    if not group:
        continue
    summary[label] = {'cases': len(group), 'v2_speedup_vs_v1': geo(r['v2_speedup_vs_v1'] for r in group),
                      'v1_over_observed_best': geo(r['v1_over_observed_best'] for r in group),
                      'v2_over_observed_best': geo(r['v2_over_observed_best'] for r in group),
                      'v2_faster_by_5pct': sum(r['v2_speedup_vs_v1'] > 1.05 for r in group),
                      'v2_slower_by_5pct': sum(r['v2_speedup_vs_v1'] < 1/1.05 for r in group),
                      'v1_within_5pct_best': sum(r['v1_over_observed_best'] <= 1.05 for r in group),
                      'v2_within_5pct_best': sum(r['v2_over_observed_best'] <= 1.05 for r in group),
                      'same_config_both_orders': sum(r['same_config_both_orders'] for r in group),
                      'geomean_v2_setup_over_v1': geo(r['v2_setup_over_v1'] for r in group),
                      'total_cost_speedups': {str(n):geo(r[f'v2_total_speedup_n{n}'] for r in group)
                                              for n in (1,100,1000,10000,100000,1000000)}}
    if metadata.get('log_model_predictions'):
        for prefix in ('v1','v2'):
            for metric in ('ratio','absolute'):
                field = prefix+'_prediction_'+metric+'_mape'
                values = [r[field] for r in group if r[field] != '']
                summary[label][field] = {'cases':len(values), 'mean':statistics.mean(values) if values else None,
                                          'median':statistics.median(values) if values else None,
                                          'over_20pct':sum(x>.2 for x in values)}
        if metadata.get('vector_candidates'):
            for prefix in ('v1','v2'):
                summary[label][prefix+'_all_width_ratio_mape'] = statistics.mean(r[prefix+'_all_width_ratio_mape'] for r in group)
                summary[label][prefix+'_all_width_absolute_mape'] = statistics.mean(r[prefix+'_all_width_absolute_mape'] for r in group)
                summary[label][prefix+'_width_within_5pct_best'] = sum(r[prefix+'_width_over_measured_best']<=1.05 for r in group)
summary['segments'] = {}
for dimension in ('nnz_bucket','v2_backend') + (('v2_reference_calibrated',) if metadata.get('log_model_predictions') else ()):
    summary['segments'][dimension] = {}
    for value in sorted({r[dimension] for r in rows}):
        group = [r for r in rows if r[dimension] == value]
        segment = {'cases':len(group), 'v2_speedup_vs_v1':geo(r['v2_speedup_vs_v1'] for r in group),
                   'v2_within_5pct_best':sum(r['v2_over_observed_best']<=1.05 for r in group)}
        if metadata.get('log_model_predictions'):
            errors = [r['v2_prediction_ratio_mape'] for r in group if r['v2_prediction_ratio_mape'] != '']
            segment['v2_prediction_ratio_mape'] = {'cases':len(errors),'mean':statistics.mean(errors) if errors else None}
            if metadata.get('vector_candidates'):
                segment['v2_all_width_ratio_mape'] = statistics.mean(r['v2_all_width_ratio_mape'] for r in group)
                segment['v2_width_within_5pct_best'] = sum(r['v2_width_over_measured_best']<=1.05 for r in group)
        summary['segments'][dimension][value] = segment
if rows:
    tmp = a.run/'v2-paired-analysis.csv.tmp'
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, a.run/'v2-paired-analysis.csv')
tmp = a.run/'v2-paired-analysis.json.tmp'
tmp.write_text(json.dumps(summary, indent=2)+'\n')
os.replace(tmp, a.run/'v2-paired-analysis.json')
print(json.dumps(summary, indent=2))
