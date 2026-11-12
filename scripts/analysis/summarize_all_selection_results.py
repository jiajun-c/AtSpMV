#!/usr/bin/env python3
"""Join H100 run status and paired timings to the complete dataset inventory."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--inventory', type=Path, required=True)
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--v2', type=Path, required=True)
p.add_argument('--outside', type=Path, required=True)
p.add_argument('--structural-audit', type=Path, required=True)
p.add_argument('--additional-scan', type=Path)
p.add_argument('--additional-outside', type=Path)
p.add_argument('--resource-retry', type=Path)
p.add_argument('--absolute-evaluation', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()

def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def sample(run, name, selection):
    status_path = run / 'status' / (name + '.json')
    if not status_path.exists():
        return {'status': 'pending'}
    status = json.loads(status_path.read_text())['status']
    if status != 'ok':
        return {'status': status}
    values = defaultdict(list)
    configs = {}
    for row in read_csv(run / 'raw' / (name + '.csv')):
        key = (row['order'], row['backend'])
        value = float(row['us'])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{run}: {name}: invalid time')
        values[key].append(value)
        configs[key] = row['config']
    orders = sorted({key[0] for key in values})
    if len(orders) != 2:
        raise ValueError(f'{run}: {name}: expected two construction orders')
    medians = {key: statistics.median(v) for key, v in values.items()}
    ratios, speedups_default, speedups_cusparse, paired_v1 = [], [], [], []
    for order in orders:
        keys = [key for key in medians if key[0] == order]
        if any((order, label) not in medians for label in (selection, 'atbsr', 'cusparse')):
            raise ValueError(f'{run}: {name}: incomplete candidates')
        ratios.append(medians[order, selection] / min(medians[key] for key in keys))
        speedups_default.append(medians[order, 'atbsr'] / medians[order, selection])
        speedups_cusparse.append(medians[order, 'cusparse'] / medians[order, selection])
        if selection == 'auto_vector_model_v2':
            if (order, 'auto_vector_model') not in medians: raise ValueError(f'{run}: {name}: missing v1 comparison')
            paired_v1.append(medians[order, 'auto_vector_model'] / medians[order, selection])
    geo = lambda v: math.exp(statistics.mean(map(math.log, v)))
    return {'status': 'ok', 'over_observed_best': math.exp(statistics.mean(map(math.log, ratios))),
            'speedup_vs_default':geo(speedups_default), 'speedup_vs_cusparse':geo(speedups_cusparse),
            'paired_v2_vs_v1':geo(paired_v1) if paired_v1 else '',
            'config': '|'.join(configs[order, selection] for order in orders)}

inventory = read_csv(a.inventory)
audited = {r['dataset']: r['status'] for r in read_csv(a.structural_audit)}
retry_index = a.resource_retry / 'index.csv' if a.resource_retry else None
retry_names = {r['dataset'] for r in read_csv(retry_index)} if retry_index and retry_index.exists() else set()
retry_evidence = {}
absolute_rows = {r['dataset']:r for r in read_csv(a.absolute_evaluation)} if a.absolute_evaluation and a.absolute_evaluation.exists() else {}
if retry_names:
    retry_evidence = {r['dataset']:r for r in json.loads((a.resource_retry/'retry-evidence.json').read_text())['retries']}
    if set(retry_evidence) != retry_names:
        raise ValueError('resource retry evidence and index differ')
records = []
for item in inventory:
    name, scope = item['dataset'], item['scope']
    audit_status = audited.get(name, '')
    scope_source = 'inventory'
    if scope == 'unknown_structure' and a.additional_scan:
        scan_status = a.additional_scan / 'status' / (name + '.json')
        if scan_status.exists():
            result = json.loads(scan_status.read_text())
            audit_status = result['status']
            if audit_status == 'ok':
                features = result['structure']
                if int(features['nnz']) > 1000000 or int(features['rows']) > 349651 or int(features['columns']) > 1785345:
                    scope, scope_source = 'outside_calibration_scope', 'additional_scan'
                else:
                    scope, scope_source = 'within_calibration_scope', 'additional_scan'
    if scope == 'unknown_structure':
        if audit_status in ('invalid_data', 'invalid_header', 'definitely_truncated', 'missing'):
            scope = 'invalid_input'
        elif audit_status == 'unsupported_int32_nnz':
            scope = 'unsupported_int32_nnz'
    row = {'dataset': name, 'scope': scope, 'scope_source': scope_source,
           'input_source': item.get('input_source','primary'), 'structural_audit_status': audit_status}
    for key, run, selection, applicable in (
        ('baseline', a.baseline, 'auto_vector_model', item['in_h100_sweep'] == 'True'),
        ('v2', a.v2, 'auto_vector_model_v2', scope == 'within_calibration_scope'),
        ('outside_refined', a.additional_outside if scope_source == 'additional_scan' and a.additional_outside else a.outside,
         'auto_model_refined', scope == 'outside_calibration_scope'),
    ):
        result = sample(run, name, selection) if applicable else {'status': 'not_applicable'}
        row[key + '_status'] = result['status']
        row[key + '_over_observed_best'] = result.get('over_observed_best', '')
        row[key + '_speedup_vs_default'] = result.get('speedup_vs_default', '')
        row[key + '_speedup_vs_cusparse'] = result.get('speedup_vs_cusparse', '')
        if key == 'v2': row['v2_paired_speedup_vs_v1'] = result.get('paired_v2_vs_v1', '')
        row[key + '_config'] = result.get('config', '')
    if a.resource_retry:
        if name in retry_names:
            retry = sample(a.resource_retry, name, 'auto_model_refined_vector')
            if retry['status'] == 'ok':
                evidence = retry_evidence[name]
                source = Path(evidence['source_run'])
                status_path = source/'status'/(name+'.json')
                log_path = source/'raw'/(name+'.log')
                if (hashlib.sha256(status_path.read_bytes()).hexdigest() != evidence['status_sha256'] or
                    hashlib.sha256(log_path.read_bytes()).hexdigest() != evidence['log_sha256']):
                    retry = {'status':'source_evidence_changed'}
                else:
                    original = json.loads(status_path.read_text()).get('matrix_sha256')
                    current = json.loads((a.resource_retry/'status'/(name+'.json')).read_text()).get('matrix_sha256')
                    if not original or original != current:
                        retry = {'status':'matrix_changed'}
        elif not retry_index.exists() and row['outside_refined_status'] in ('pending','error','timeout'):
            retry = {'status':'pending_classification'}
        else:
            retry = {'status':'not_applicable'}
        row['resource_retry_status'] = retry['status']
        row['resource_retry_over_observed_best'] = retry.get('over_observed_best', '')
        row['resource_retry_speedup_vs_default'] = retry.get('speedup_vs_default', '')
        row['resource_retry_speedup_vs_cusparse'] = retry.get('speedup_vs_cusparse', '')
        row['resource_retry_config'] = retry.get('config', '')
    if a.absolute_evaluation:
        reference = absolute_rows.get(name)
        v2_status = row['v2_status']
        row['absolute_reference_status'] = ('ok' if reference and v2_status == 'ok' else
            'pending' if v2_status == 'pending' else 'not_evaluated' if v2_status == 'ok' else 'not_applicable')
        for field in ('split','predicted_reference_us','actual_reference_us','old_reference_mape','new_reference_mape','old_vector_absolute_mape','new_vector_absolute_mape'):
            row['absolute_reference_'+field] = reference[field] if reference and v2_status == 'ok' else ''
    records.append(row)

a.output.mkdir(parents=True, exist_ok=True)
temporary_csv = a.output / 'all-datasets.csv.tmp'
with temporary_csv.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys()); writer.writeheader(); writer.writerows(records)
os.replace(temporary_csv, a.output / 'all-datasets.csv')
summary = {'total': len(records), 'scope_counts': dict(Counter(r['scope'] for r in records)),
           'method': 'Each ratio compares actual selected Plan to the fastest measured candidate separately in each construction order, then takes their geometric mean. Candidate sets differ between runs. Timings are warm Graph execution and exclude setup. Pending and failed cases are never included in timing metrics.'}
for key in ('baseline', 'v2', 'outside_refined') + (('resource_retry',) if a.resource_retry else ()):
    numbers = [r[key + '_over_observed_best'] for r in records if r[key + '_status'] == 'ok']
    stats = {'status_counts': dict(Counter(r[key + '_status'] for r in records))}
    if numbers:
        stats.update(cases=len(numbers), geomean_over_observed_best=math.exp(statistics.mean(map(math.log, numbers))),
                     within_5pct=sum(x <= 1.05 for x in numbers), over_10pct=sum(x > 1.1 for x in numbers))
        for label in ('speedup_vs_default','speedup_vs_cusparse'):
            values = [r[key+'_'+label] for r in records if r[key+'_status'] == 'ok']
            stats['geomean_'+label] = math.exp(statistics.mean(map(math.log, values)))
        if key == 'v2':
            values = [r['v2_paired_speedup_vs_v1'] for r in records if r['v2_status'] == 'ok']
            stats['geomean_paired_speedup_vs_v1'] = math.exp(statistics.mean(map(math.log, values)))
    summary[key] = stats
if a.absolute_evaluation:
    group = [r for r in records if r['absolute_reference_status'] == 'ok']
    result = {'status_counts':dict(Counter(r['absolute_reference_status'] for r in records))}
    if group:
        result['cases'] = len(group)
        for field in ('old_reference_mape','new_reference_mape','old_vector_absolute_mape','new_vector_absolute_mape'):
            result[field] = statistics.mean(float(r['absolute_reference_'+field]) for r in group)
    summary['absolute_reference'] = result
temporary_json = a.output / 'summary.json.tmp'
temporary_json.write_text(json.dumps(summary, indent=2) + '\n')
os.replace(temporary_json, a.output / 'summary.json')
print(json.dumps(summary, indent=2))
