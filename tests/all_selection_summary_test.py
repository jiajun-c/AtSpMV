#!/usr/bin/env python3
"""Exercise coverage and paired-oracle aggregation on a small fixture."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as temporary:
    d = Path(temporary)
    def write_csv(path, rows):
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    inventory = [
        {'dataset': 'one', 'scope': 'within_calibration_scope', 'in_h100_sweep': 'True'},
        {'dataset': 'two', 'scope': 'outside_calibration_scope', 'in_h100_sweep': 'False'},
        {'dataset': 'three', 'scope': 'unknown_structure', 'in_h100_sweep': 'False'},
        {'dataset': 'four', 'scope': 'unknown_structure', 'in_h100_sweep': 'False'},
    ]
    write_csv(d/'inventory.csv', inventory)
    write_csv(d/'audit.csv', [{'dataset': 'three', 'status': 'invalid_data'}, {'dataset':'four', 'status':'unsupported_int32_nnz'}])
    for name in ('base', 'v2', 'outside', 'extra', 'scan', 'retry'):
        (d/name/'status').mkdir(parents=True)
        (d/name/'raw').mkdir()
    (d/'base/status/one.json').write_text('{"status":"ok"}')
    records = []
    for order in ('baseline_first', 'selection_first'):
        for backend, us in (('atbsr', 2), ('auto_vector_model', 1), ('cusparse', 1.5)):
            records.append({'order': order, 'backend': backend, 'us': us, 'config': backend})
    write_csv(d/'base/raw/one.csv', records)
    (d/'outside/status/two.json').write_text('{"status":"error"}')
    (d/'scan/status/three.json').write_text(json.dumps({'status':'ok', 'structure':{'nnz': '2000000', 'rows': '1', 'columns': '1'}}))
    cmd = ['python3', str(root/'scripts/analysis/summarize_all_selection_results.py'),
           '--inventory', str(d/'inventory.csv'), '--baseline', str(d/'base'), '--v2', str(d/'v2'),
           '--outside', str(d/'outside'), '--structural-audit', str(d/'audit.csv'),
           '--additional-scan', str(d/'scan'), '--additional-outside', str(d/'extra'),
           '--resource-retry', str(d/'retry'), '--absolute-evaluation', str(d/'reference.csv'), '--output', str(d/'out')]
    subprocess.run(cmd, check=True, capture_output=True)
    result = json.loads((d/'out/summary.json').read_text())
    assert result['total'] == 4
    assert result['scope_counts']['unsupported_int32_nnz'] == 1
    assert result['baseline']['cases'] == 1 and result['baseline']['geomean_over_observed_best'] == 1
    assert result['outside_refined']['status_counts']['error'] == 1
    rows = {r['dataset']: r for r in csv.DictReader((d/'out/all-datasets.csv').open())}
    assert rows['three']['structural_audit_status'] == 'ok'
    assert rows['three']['scope'] == 'outside_calibration_scope' and rows['three']['outside_refined_status'] == 'pending'
    assert rows['one']['v2_status'] == 'pending'
    assert rows['two']['resource_retry_status'] == 'pending_classification'
    (d/'outside/raw/two.log').write_text('out of memory\n')
    source_status = d/'outside/status/two.json'
    source_status.write_text(json.dumps({'status':'error','matrix_sha256':'original'}))
    evidence = {'dataset':'two','source_run':str(d/'outside'),
                'status_sha256':hashlib.sha256(source_status.read_bytes()).hexdigest(),
                'log_sha256':hashlib.sha256((d/'outside/raw/two.log').read_bytes()).hexdigest()}
    (d/'retry/retry-evidence.json').write_text(json.dumps({'retries':[evidence]}))
    write_csv(d/'retry/index.csv', [{'dataset':'two'}])
    subprocess.run(cmd,check=True,capture_output=True)
    rows = {r['dataset']: r for r in csv.DictReader((d/'out/all-datasets.csv').open())}
    assert rows['two']['resource_retry_status'] == 'pending'
    retry_records=[]
    for order in ('baseline_first','selection_first'):
        for backend in ('auto_model_refined_vector','atbsr','cusparse'):
            retry_records.append({'order':order,'backend':backend,'us':1,'config':backend})
    write_csv(d/'retry/raw/two.csv', retry_records)
    (d/'retry/status/two.json').write_text(json.dumps({'status':'ok','matrix_sha256':'changed'}))
    subprocess.run(cmd,check=True,capture_output=True)
    rows = {r['dataset']: r for r in csv.DictReader((d/'out/all-datasets.csv').open())}
    assert rows['two']['resource_retry_status'] == 'matrix_changed'
    (d/'outside/raw/two.log').write_text('modified\n')
    subprocess.run(cmd,check=True,capture_output=True)
    rows = {r['dataset']: r for r in csv.DictReader((d/'out/all-datasets.csv').open())}
    assert rows['two']['resource_retry_status'] == 'source_evidence_changed'
    records=[]
    for order in ('baseline_first','selection_first'):
        for backend, us in (('atbsr',3),('cusparse',4),('auto_vector_model',2),('auto_vector_model_v2',1)):
            records.append({'order':order,'backend':backend,'us':us,'config':backend})
    write_csv(d/'v2/raw/one.csv', records)
    (d/'v2/status/one.json').write_text('{"status":"ok"}')
    subprocess.run(cmd,check=True,capture_output=True)
    data=json.loads((d/'out/summary.json').read_text())
    assert data['v2']['geomean_paired_speedup_vs_v1'] == 2
    assert abs(data['v2']['geomean_speedup_vs_default']-3) < 1e-12
    write_csv(d/'reference.csv',[{'dataset':'one','split':'holdout','predicted_reference_us':'2','actual_reference_us':'3',
        'old_reference_mape':'.5','new_reference_mape':'.1','old_vector_absolute_mape':'.4','new_vector_absolute_mape':'.2'}])
    subprocess.run(cmd,check=True,capture_output=True)
    data=json.loads((d/'out/summary.json').read_text())
    assert data['absolute_reference']['cases']==1 and data['absolute_reference']['new_reference_mape']==.1
print('all-dataset selection summary passed')
