#!/usr/bin/env python3
"""Validate native AtBSR grouping against cuSPARSE with default L2 policy."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

from benchmark_mfem_heat import write_json

BACKENDS = ('cusparse', 'atbsr', 'atbsr-grouped')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(a):
    records = {}; trials = []
    for rnd in range(a.rounds):
        folder = a.output/f'round-{rnd}'
        manifest = json.loads((folder/'manifest.json').read_text())
        if manifest['executable_sha256'] != digest(a.exe) or manifest['persist_x']:
            raise ValueError('wrong executable or external cache policy')
        ref = np.fromfile(folder/'cusparse-0.bin', dtype=np.float64)
        baseline = json.loads((folder/'cusparse-0.json').read_text())['result']
        for backend in BACKENDS:
            rec = json.loads((folder/f'{backend}-0.json').read_text()); r = rec['result']; cmd = rec['command']
            if rec['returncode'] or r['steps'] != a.steps or '-persist-x' in cmd or '-dump-t' in cmd:
                raise ValueError('failed process, wrong steps, cache policy or timing')
            if cmd[cmd.index('-b')+1] != backend or cmd[cmd.index('-a')+1] != '0.01' or '-fa' not in cmd:
                raise ValueError('different application configuration')
            if r['cg_iterations'] != baseline['cg_iterations'] or r['spmv_calls'] != baseline['spmv_calls']:
                raise ValueError('different solver workload')
            field = np.fromfile(folder/f'{backend}-0.bin', dtype=np.float64)
            if field.shape != ref.shape or len(field) != r['dofs'] or not np.isfinite(field).all():
                raise ValueError('incomplete field')
            error = float(np.max(np.abs(field-ref)))/max(1., float(np.max(np.abs(ref))))
            if error > 1e-7 or r['mass_relative_error'] > 1e-6 or r['final_energy'] > r['initial_energy']*(1+1e-6):
                raise ValueError('incorrect physical solution')
            log = (folder/f'{backend}-0.log').read_text()
            if '--no-persist-x' not in log:
                raise ValueError('L2 policy was not confirmed disabled')
            if backend == 'atbsr-grouped' and log.count('grouped_columns=1') != 3:
                raise ValueError('new layout was not actually applied')
            if backend == 'atbsr' and 'grouped_columns=1' in log:
                raise ValueError('baseline was changed')
            row = {'round': rnd, 'backend': backend, 'dofs': r['dofs'], 'steps': r['steps'],
                   'cg_iterations': r['cg_iterations'], 'spmv_calls': r['spmv_calls'], 'field_error': error,
                   **{k: r[k] for k in ('total_s', 'linear_s', 'plan_s', 'initialization_s', 'evolution_s')},
                   'process_wall_s': rec['process_wall_s']}
            for key in ('total_s', 'linear_s', 'plan_s'):
                if not math.isfinite(row[key]) or row[key] <= 0:
                    raise ValueError('invalid timing')
            records[rnd, backend] = row; trials.append(row)
    summary = {'validated_processes': len(trials), 'cpu_affinity': a.cpu, 'persist_x': False,
               'max_field_error': max(r['field_error'] for r in trials), 'backends': {}}
    for backend in BACKENDS:
        rows = [records[r, backend] for r in range(a.rounds)]
        result = {'median_'+k: statistics.median(r[k] for r in rows)
                  for k in ('total_s', 'linear_s', 'plan_s', 'initialization_s', 'evolution_s', 'process_wall_s')}
        for reference in ('cusparse', 'atbsr'):
            ratios = {}
            for key in ('total_s', 'linear_s', 'process_wall_s'):
                logs = np.log([records[r, reference][key]/records[r, backend][key] for r in range(a.rounds)])
                samples = np.random.default_rng(20261104).integers(0, a.rounds, (100000, a.rounds))
                ratios[key] = {'geomean_speedup': float(np.exp(logs.mean())),
                               'paired_bootstrap_95': np.exp(np.quantile(logs[samples].mean(axis=1), [.025, .975])).tolist(),
                               'wins': int(np.sum(logs > 0)), 'rounds': a.rounds}
            result['vs_'+reference] = ratios
        summary['backends'][backend] = result
    write_json(a.output/'summary.json', summary)
    with (a.output/'trials.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(trials[0])); writer.writeheader(); writer.writerows(trials)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('exe', 'mesh', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--gpu', type=int, default=3)
    p.add_argument('--cpu', type=int, default=30)
    p.add_argument('--refine', type=int, default=7)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--rounds', type=int, default=8)
    p.add_argument('--analyze-only', action='store_true')
    a = p.parse_args()
    a.exe = a.exe.resolve(); a.mesh = a.mesh.resolve(); a.output = a.output.resolve()
    if a.rounds < 2 or a.cpu not in os.sched_getaffinity(0):
        p.error('need >=2 rounds and an available CPU affinity')
    a.output.mkdir(parents=True, exist_ok=True)
    helper = Path(__file__).with_name('benchmark_mfem_heat.py')
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items() if k != 'analyze_only'}
    manifest.update(executable_sha256=digest(a.exe), mesh_sha256=digest(a.mesh), runner_sha256=digest(__file__),
                    helper_sha256=digest(helper), persist_x=False, prefetch=2, warps=4)
    path = a.output/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('experiment identity changed; use a new directory')
    write_json(path, manifest)
    if not a.analyze_only:
        os.sched_setaffinity(0, {a.cpu})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        orders = [BACKENDS, (BACKENDS[1], BACKENDS[2], BACKENDS[0]), (BACKENDS[2], BACKENDS[0], BACKENDS[1])]
        for rnd in range(a.rounds):
            while int(subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=utilization.gpu',
                                               '--format=csv,noheader,nounits'], text=True).strip()) > 5:
                print('waiting for GPU idle', flush=True); time.sleep(5)
            order = orders[(rnd//2) % 3]
            if rnd % 2:
                order = order[::-1]
            folder = a.output/f'round-{rnd}'; folder.mkdir(exist_ok=True)
            cmd = [sys.executable, str(helper), '--exe', str(a.exe), '--mesh', str(a.mesh), '--output', str(folder),
                   '--refine', str(a.refine), '--steps', str(a.steps), '--alpha', '0.01', '--rounds', '1',
                   '--backends', ','.join(order), '--prefetch', '2', '--warps', '4', '--gpu-assembly', '--timeout', '120']
            with (folder/'runner.log').open('a') as log:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
            print(f'completed round {rnd+1}/{a.rounds}: {order}', flush=True)
    analyze(a)
    write_json(a.output/'completed.json', {'status': 'complete', 'validated_processes': a.rounds*3})


if __name__ == '__main__':
    main()
