#!/usr/bin/env python3
"""Replay a predeclared prior-matrix sample with original and grouped AtBSR."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import subprocess
import time

from benchmark_dsm_atbsr_full import apps, sha, write_json

NATIVE = ('atbsr_historical', 'atbsr', 'atbsr_m32', 'atbsr_compressed', 'atbsr_grouped', 'atbsr_grouped_model')
EXPECTED = NATIVE + ('cusparse', 'cusparse_default', 'cusparse_alg2')
COMPARISONS = {'group_vs_m32': ('atbsr_m32','atbsr_grouped'),
               'group_vs_original': ('atbsr','atbsr_grouped'),
               'model_vs_original': ('atbsr','atbsr_grouped_model'),
               'group_vs_old_compression': ('atbsr_compressed','atbsr_grouped'),
               'group_vs_cusparse': ('cusparse','atbsr_grouped'),
               'model_vs_cusparse': ('cusparse','atbsr_grouped_model'),
               'original_vs_cusparse': ('cusparse','atbsr'),
               'group_vs_historical': ('atbsr_historical','atbsr_grouped')}


def analyze(a, selection):
    rows = []; failures = []
    for item in selection['matrices']:
        name = item['dataset']; path = a.output/'status'/(name+'.json')
        if not path.exists():
            failures.append({'dataset': name, 'status': 'pending'}); continue
        status = json.loads(path.read_text())
        if status['status'] != 'ok':
            failures.append({'dataset': name, 'status': status['status']}); continue
        orders = []; layouts = []
        for order in (0, 1):
            stem = a.output/'raw'/f'{name}-{order}'
            raw = list(csv.DictReader(stem.with_suffix('.csv').open()))
            times = {b: [] for b in EXPECTED}
            for r in raw:
                value = float(r['us'])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError('invalid timing')
                times[r['backend']].append(value)
            if any(len(t) != a.trials for t in times.values()):
                raise ValueError('incomplete trial count')
            orders.append({b: statistics.median(t) for b, t in times.items()})
            layout = {}
            for line in stem.with_suffix('.log').read_text().splitlines():
                if line.startswith('PLAN_LAYOUT,'):
                    p = line.split(','); layout[p[1]] = dict(zip(
                        ('rows','columns','nnz','tile','balanced','grouped','compressed','column_bytes'), map(int, p[2:])))
            if set(layout) != set(NATIVE):
                raise ValueError('missing layout metadata')
            layouts.append(layout)
        if layouts[0] != layouts[1]:
            raise ValueError('configuration changed between construction orders')
        gmean = lambda values: math.exp(statistics.mean(math.log(v) for v in values))
        original = layouts[0]['atbsr']; grouped = layouts[0]['atbsr_grouped']; model = layouts[0]['atbsr_grouped_model']
        row = {'dataset': name, **{k: original[k] for k in ('rows','columns','nnz')},
               'prior_fastest_fixed': item['prior_fastest_fixed'],
               'original_shape': f"{1<<original['tile']}x{32>>original['tile']}", 'original_balanced': original['balanced'],
               'grouped_active': grouped['grouped'], 'grouped_model_active': model['grouped'],
               'grouped_column_bytes': grouped['column_bytes']}
        for b in EXPECTED:
            row[b+'_us'] = gmean(o[b] for o in orders)
        for key, (base, candidate) in COMPARISONS.items():
            row[key] = gmean(o[base]/o[candidate] for o in orders)
        row['group_vs_best_cusparse'] = gmean(min(o[b] for b in ('cusparse','cusparse_default','cusparse_alg2'))/o['atbsr_grouped'] for o in orders)
        rows.append(row)
    summary = {'requested': len(selection['matrices']), 'successful': len(rows), 'failures': failures,
               'scope': 'Predeclared 24-matrix prior sample on one H100. Two independent processes with opposite construction order; interleaved warm Graph timings. No persisting L2. Setup excluded.',
               'groups': {}}
    for name, subset in [('all', rows), ('grouped_active', [r for r in rows if r['grouped_active']]),
                         ('grouped_model_active', [r for r in rows if r['grouped_model_active']])]:
        result = {'matrices': len(subset)}
        for metric in list(COMPARISONS) + ['group_vs_best_cusparse']:
            if subset:
                result[metric] = {'geomean': math.exp(statistics.mean(math.log(r[metric]) for r in subset)),
                                  'wins_over_1pct': sum(r[metric] > 1.01 for r in subset),
                                  'losses_over_1pct': sum(r[metric] < 1/1.01 for r in subset)}
        summary['groups'][name] = result
    if rows:
        with (a.output/'analysis.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(a.output/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=3); p.add_argument('--trials', type=int, default=15)
    p.add_argument('--timeout', type=int, default=300); p.add_argument('--analyze-only', action='store_true')
    a = p.parse_args(); a.output = a.output.resolve(); a.exe = a.exe.resolve()
    selection = json.loads((a.output/'selection.json').read_text())
    for folder in ('raw','status','provenance'):
        (a.output/folder).mkdir(exist_ok=True)
    uuid = subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'], text=True).strip()
    metadata = {'executable': str(a.exe), 'exe_sha256': sha(a.exe), 'selection_sha256': sha(a.output/'selection.json'),
                'runner_sha256': sha(Path(__file__)), 'gpu_uuid': uuid, 'trials': a.trials, 'timeout': a.timeout,
                'prefetch': 2, 'historical_prefetch': 0, 'warps': 4, 'persist_x': False,
                'configs_sha256': [sha(a.output/n) for n in ('configs.csv','configs-reverse.csv')]}
    path = a.output/'metadata.json'
    if path.exists() and json.loads(path.read_text()) != metadata:
        raise ValueError('experiment identity changed')
    write_json(path, metadata)
    if not a.analyze_only:
        env = {k: v for k, v in os.environ.items() if not k.startswith(('ATSPMV_MODEL_', 'ATSPMV_BENCH_'))}
        env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   ATSPMV_TUNE_PREFETCH='2', ATSPMV_TUNE_WARPS='4', ATSPMV_TUNE_CAP='8', ATSPMV_TUNE_CTA_REDUCE='0',
                   ATSPMV_BENCH_CUSPARSE_ALGORITHMS='1', ATSPMV_BENCH_INTERNAL_GRAPH_EVENTS='1')
        order = list(selection['matrices']); random.Random(20261105).shuffle(order)
        for ordinal, item in enumerate(order):
            name = item['dataset']; status_path = a.output/'status'/(name+'.json')
            if status_path.exists() and json.loads(status_path.read_text()).get('status') == 'ok':
                continue
            matrix = Path(item['path'])
            if not matrix.is_file():
                write_json(status_path, {'status':'missing','dataset':name}); continue
            record = {'dataset':name,'matrix_sha256':sha(matrix),'status':'running','orders':[]}
            write_json(status_path, record)
            for construction in (0, 1):
                while apps(uuid):
                    print('waiting for unused GPU', flush=True); time.sleep(5)
                stem = a.output/'raw'/f'{name}-{construction}'
                config = a.output/('configs.csv' if construction == 0 else 'configs-reverse.csv')
                cmd = [str(a.exe), str(matrix), str(config), str(20261105+ordinal*2+construction), str(a.trials)]
                with stem.with_suffix('.csv').open('w') as out, stem.with_suffix('.log').open('w') as err:
                    start = time.monotonic()
                    proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=err, start_new_session=True)
                    reason = None
                    while proc.poll() is None:
                        foreign = [r for r in apps(uuid) if r['pid'] != proc.pid]
                        if foreign or time.monotonic()-start > a.timeout:
                            reason = 'interference' if foreign else 'timeout'
                            os.killpg(proc.pid, signal.SIGTERM)
                            try: proc.wait(timeout=3)
                            except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                            break
                        time.sleep(.5)
                record['orders'].append({'command':cmd,'returncode':proc.returncode,'elapsed_s':time.monotonic()-start,'reason':reason})
                if reason or proc.returncode:
                    record['status'] = reason or 'failed'; break
            else:
                record['status'] = 'ok'
            write_json(status_path, record)
            print(f"{ordinal+1}/24 {name}: {record['status']}", flush=True)
        shutil.copyfile(__file__, a.output/'provenance'/Path(__file__).name)
    analyze(a, selection)
    write_json(a.output/'completed.json', {'status':'complete','note':'Inspect summary failures; completion does not imply every matrix succeeded.'})


if __name__ == '__main__':
    main()
