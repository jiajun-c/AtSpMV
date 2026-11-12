#!/usr/bin/env python3
"""Paired 2x2 ablation of lossless tile columns and persisting X in MFEM."""
import argparse
import csv
import hashlib
import itertools
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


BACKENDS = ('cusparse', 'atbsr', 'atbsr-compressed')
CASES = ((5, 100), (7, 20))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wait_idle(gpu):
    while True:
        utilization = int(subprocess.check_output(
            ['nvidia-smi', '-i', str(gpu), '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip())
        if utilization <= 5:
            return
        print(f'GPU {gpu} utilization={utilization}%, waiting for idle', flush=True)
        time.sleep(5)


def run_applications(a):
    orders = list(itertools.permutations(BACKENDS))
    script = Path(__file__).with_name('benchmark_mfem_heat.py')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for rnd in range(a.rounds):
        cases = CASES if rnd % 2 == 0 else CASES[::-1]
        for refine, steps in cases:
            for persist in ((0, 1) if rnd % 2 == 0 else (1, 0)):
                folder = a.output/f'r{refine}'/f'round-{rnd}'/f'persist-{persist}'
                order = orders[(rnd+persist*3) % len(orders)]
                wait_idle(a.gpu)
                cmd = [sys.executable, str(script), '--exe', str(a.exe), '--mesh', str(a.mesh),
                       '--output', str(folder), '--refine', str(refine), '--steps', str(steps),
                       '--alpha', '0.01', '--rounds', '1', '--backends', ','.join(order),
                       '--prefetch', '2', '--warps', '4', '--gpu-assembly', '--timeout', '120']
                if persist:
                    cmd.append('--persist-x')
                folder.mkdir(parents=True, exist_ok=True)
                with (folder/'runner.log').open('a') as log:
                    subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=420)
                print(f'completed r{refine} round={rnd} persist={persist} order={order}', flush=True)


def audit(a):
    rows = []
    records = {}
    for refine, steps in CASES:
        for rnd in range(a.rounds):
            for persist in (0, 1):
                folder = a.output/f'r{refine}'/f'round-{rnd}'/f'persist-{persist}'
                manifest = json.loads((folder/'manifest.json').read_text())
                if manifest['executable_sha256'] != sha(a.exe) or manifest['prefetch'] != 2 or manifest['warps'] != 4:
                    raise ValueError('changed executable or kernel configuration')
                baseline = json.loads((folder/'cusparse-0.json').read_text())['result']
                field_ref = np.fromfile(folder/'cusparse-0.bin', dtype=np.float64)
                for backend in BACKENDS:
                    rec = json.loads((folder/f'{backend}-0.json').read_text())
                    r = rec['result']
                    command = rec['command']
                    expected = {'-r': str(refine), '-b': backend, '-s': '21', '-o': '2', '-a': '0.01', '-dt': '0.01', '-tf': str(steps*.01)}
                    if any(command[command.index(k)+1] != v for k, v in expected.items()):
                        raise ValueError('unequal application settings')
                    if ('-persist-x' in command) != bool(persist) or '-fa' not in command:
                        raise ValueError('unexpected cache or assembly settings')
                    if any(k in command for k in ('-rebuild', '-rebuild-t', '-host-coeff', '-dump-t')):
                        raise ValueError('unexpected ablation setting')
                    if rec['returncode'] or r['steps'] != steps or r['cg_iterations'] != baseline['cg_iterations'] or r['spmv_calls'] != baseline['spmv_calls']:
                        raise ValueError('incomplete application or unequal iteration counts')
                    field = np.fromfile(folder/f'{backend}-0.bin', dtype=np.float64)
                    if len(field) != r['dofs'] or field.shape != field_ref.shape or not np.isfinite(field).all():
                        raise ValueError('invalid complete field')
                    error = float(np.max(np.abs(field-field_ref)))/max(1., float(np.max(np.abs(field_ref))))
                    if error > 1e-7 or not math.isfinite(r['mass_relative_error']) or r['mass_relative_error'] > 1e-6 or r['final_energy'] > r['initial_energy']*(1+1e-6):
                        raise ValueError('numerical validation failed')
                    if backend != 'cusparse':
                        logs = (folder/f'{backend}-0.log').read_text().splitlines()
                        plans = [line for line in logs if line.startswith('plan rows=')]
                        if len(plans) != 3 or any('tile=5 balanced=0 ' not in line for line in plans):
                            raise ValueError('AtBSR shape/mode changed: expected normal 32x1')
                        encoded = [line for line in logs if line.startswith('compressed_columns bytes=')]
                        if len(encoded) != (3 if backend == 'atbsr-compressed' else 0):
                            raise ValueError('column compression was not actually applied')
                    for key in ('total_s', 'linear_s', 'plan_s', 'evolution_s'):
                        if not math.isfinite(r[key]) or r[key] <= 0:
                            raise ValueError('invalid timing')
                    row = {'refine': refine, 'round': rnd, 'persist_x': persist, 'backend': backend,
                           **{k: r[k] for k in ('dofs', 'nnz', 'steps', 'cg_iterations', 'total_s', 'linear_s', 'plan_s', 'evolution_s')},
                           'process_wall_s': rec['process_wall_s'], 'field_relative_error': error}
                    records[refine, rnd, persist, backend] = row
                    rows.append(row)
    return rows, records


def paired_ratio(numerator, denominator, key):
    logs = np.log([x[key]/y[key] for x, y in zip(numerator, denominator)])
    indices = np.random.default_rng(20261104).integers(0, len(logs), (100000, len(logs)))
    return {'geomean_speedup': float(np.exp(logs.mean())),
            'paired_bootstrap_95': np.exp(np.quantile(logs[indices].mean(axis=1), [.025, .975])).tolist(),
            'wins': int(np.sum(logs > 0)), 'rounds': len(logs)}


def analyze(a):
    rows, records = audit(a)
    summary = {'cells': [], 'compression_effect': [], 'cache_effect': [],
               'validated_processes': len(rows), 'max_field_relative_error': max(r['field_relative_error'] for r in rows),
               'scope': 'Two MFEM scales of one star.mesh problem, not a cross-matrix benchmark. Ratios are paired geometric means; intervals resample repeated rounds only.'}
    for refine, _ in CASES:
        for persist in (0, 1):
            for backend in BACKENDS:
                subset = [records[refine, r, persist, backend] for r in range(a.rounds)]
                summary['cells'].append({'refine': refine, 'persist_x': persist, 'backend': backend,
                                         **{k: subset[0][k] for k in ('dofs', 'nnz', 'steps')},
                                         **{'median_'+k: statistics.median(r[k] for r in subset) for k in ('total_s', 'linear_s', 'plan_s', 'process_wall_s')}})
            baseline = [records[refine, r, persist, 'atbsr'] for r in range(a.rounds)]
            compressed = [records[refine, r, persist, 'atbsr-compressed'] for r in range(a.rounds)]
            summary['compression_effect'].append({'refine': refine, 'persist_x': persist,
                **{k: paired_ratio(baseline, compressed, k) for k in ('total_s', 'linear_s', 'plan_s', 'process_wall_s')}})
        for backend in BACKENDS:
            off = [records[refine, r, 0, backend] for r in range(a.rounds)]
            on = [records[refine, r, 1, backend] for r in range(a.rounds)]
            summary['cache_effect'].append({'refine': refine, 'backend': backend,
                **{k: paired_ratio(off, on, k) for k in ('total_s', 'linear_s')}})
    report = a.output/'report'; report.mkdir(exist_ok=True)
    write_json(report/'summary.json', summary)
    for name, data in [('trials', rows), ('cells', summary['cells'])]:
        with (report/(name+'.csv')).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (refine, _) in zip(axes, CASES):
        subset = [r for r in summary['cells'] if r['refine'] == refine]
        for i, backend in enumerate(BACKENDS):
            values = [next(r['median_total_s'] for r in subset if r['persist_x'] == p and r['backend'] == backend) for p in (0, 1)]
            ax.bar(np.arange(2)+(i-1)*.25, values, width=.25, label=backend)
        ax.set_xticks([0, 1], ['Default cache', 'Persisting X'])
        ax.set_ylabel('Application compute time (s)')
        ax.set_title(f"{subset[0]['dofs']:,} DoFs, {subset[0]['steps']} steps")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(report/'ablation.png', dpi=200); fig.savefig(report/'ablation.pdf'); plt.close(fig)
    print(json.dumps(summary['compression_effect'], indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('exe', 'mesh', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--gpu', type=int, default=3)
    p.add_argument('--rounds', type=int, default=8)
    p.add_argument('--analyze-only', action='store_true')
    a = p.parse_args()
    if a.rounds < 2:
        p.error('at least two rounds are required')
    a.exe = a.exe.resolve(); a.mesh = a.mesh.resolve(); a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = {'exe': str(a.exe), 'mesh': str(a.mesh), 'gpu': a.gpu, 'rounds': a.rounds,
                'cases': CASES, 'executable_sha256': sha(a.exe), 'mesh_sha256': sha(a.mesh),
                'prefetch': 2, 'warps': 4, 'tile': '32x1 normal', 'precision': 'FP64',
                'runner_sha256': sha(__file__), 'benchmark_sha256': sha(Path(__file__).with_name('benchmark_mfem_heat.py'))}
    # JSON round-trip normalizes case tuples before strict resume comparison.
    manifest = json.loads(json.dumps(manifest))
    path = a.output/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('experiment identity changed: use a new output directory')
    write_json(path, manifest)
    if not a.analyze_only:
        run_applications(a)
    analyze(a)
    write_json(a.output/'completed.json', {'status': 'complete', 'validated_processes': a.rounds*12})


if __name__ == '__main__':
    main()
