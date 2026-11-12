#!/usr/bin/env python3
"""Audit corrected MFEM trials and export application-level comparisons."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np


ORDER = ('mfem-rebuild', 'mfem-hostcoeff', 'mfem', 'cusparse', 'atbsr', 'mixed')


def audit(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    exe = Path(manifest['exe'])
    if hashlib.sha256(exe.read_bytes()).hexdigest() != manifest['executable_sha256']:
        raise ValueError(f'executable hash mismatch: {folder}')
    records = {}
    rows = []
    for rnd in range(manifest['rounds']):
        for backend in ORDER:
            rec = json.loads((folder/f'{backend}-{rnd}.json').read_text())
            result = rec.get('result') or {}
            if rec.get('returncode') != 0 or result.get('steps') != manifest['steps']:
                raise ValueError(f'failed or incomplete trial: {folder} {backend} {rnd}')
            if not all(math.isfinite(result[k]) and result[k] > 0 for k in ('total_s', 'evolution_s')):
                raise ValueError('invalid timing')
            if result['mass_relative_error'] > 1e-6 or result['final_energy'] > result['initial_energy']*(1+1e-6):
                raise ValueError('physical invariant failed')
            command = rec['command']
            if command[command.index('-s')+1] != '21' or command[command.index('-a')+1] != str(manifest['alpha']):
                raise ValueError('unexpected physical configuration')
            if backend == 'mfem-rebuild' and not all(k in command for k in ('-rebuild', '-rebuild-t', '-host-coeff')):
                raise ValueError('rebuild reference did not use the intended configuration')
            if backend in ('atbsr', 'mixed', 'cusparse', 'mfem') and any(k in command for k in ('-rebuild', '-rebuild-t', '-host-coeff')):
                raise ValueError('optimized trial has unexpected reference flags')
            field = np.fromfile(folder/f'{backend}-{rnd}.bin', dtype=np.float64)
            if len(field) != result['dofs'] or not np.isfinite(field).all():
                raise ValueError('incomplete or invalid field')
            records[rnd, backend] = (result, field)
        reference, field_ref = records[rnd, 'cusparse']
        for backend in ORDER:
            result, field = records[rnd, backend]
            if field.shape != field_ref.shape or result['nnz'] != reference['nnz']:
                raise ValueError('incomparable matrix or field dimensions')
            error = float(np.max(np.abs(field-field_ref))) / max(1., float(np.max(np.abs(field_ref))))
            if error > 1e-7:
                raise ValueError('full-field comparison failed')
            rows.append({'case': folder.name, 'round': rnd, 'backend': backend, 'dofs': result['dofs'],
                         'nnz': result['nnz'], 'steps': result['steps'], 'total_s': result['total_s'],
                         'evolution_s': result['evolution_s'], 'linear_s': result['linear_s'],
                         'plan_s': result['plan_s'], 'cg_iterations': result['cg_iterations'],
                         'relative_field_error': error, 'mass_relative_error': result['mass_relative_error']})
    summary = []
    for backend in ORDER:
        subset = [r for r in rows if r['backend'] == backend]
        paired = [records[r, 'mfem-rebuild'][0]['total_s']/records[r, backend][0]['total_s'] for r in range(manifest['rounds'])]
        cu_paired = [records[r, 'cusparse'][0]['total_s']/records[r, backend][0]['total_s'] for r in range(manifest['rounds'])]
        summary.append({'case': folder.name, 'backend': backend, 'dofs': subset[0]['dofs'], 'nnz': subset[0]['nnz'],
                        'steps': subset[0]['steps'], 'trials': len(subset),
                        'median_total_s': statistics.median(r['total_s'] for r in subset),
                        'min_total_s': min(r['total_s'] for r in subset), 'max_total_s': max(r['total_s'] for r in subset),
                        'median_evolution_s': statistics.median(r['evolution_s'] for r in subset),
                        'median_linear_s': statistics.median(r['linear_s'] for r in subset),
                        'geomean_vs_rebuild': math.exp(statistics.mean(map(math.log, paired))),
                        'geomean_vs_cusparse_reuse': math.exp(statistics.mean(map(math.log, cu_paired))),
                        'max_relative_field_error': max(r['relative_field_error'] for r in subset)})
    return rows, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/applications/mfem-heat'))
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    output = args.output or args.root/'report'
    output.mkdir(parents=True, exist_ok=True)
    trials, summary = [], []
    for case in ('coherent-r5', 'coherent-r7'):
        a, b = audit(args.root/case)
        trials.extend(a);summary.extend(b)
    for name, data in (('trials.csv', trials), ('summary.csv', summary)):
        with (output/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader();writer.writerows(data)
    report = {'validated_processes': len(trials), 'max_relative_field_error': max(r['relative_field_error'] for r in trials),
              'max_mass_relative_error': max(r['mass_relative_error'] for r in trials), 'summary': summary,
              'timing': 'Application compute wall time includes initialization, excludes final validation, output and teardown; speedups are paired geometric means.'}
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['MFEM\nrebuild', 'MFEM reuse\nscalar coeff.', 'MFEM reuse\nbulk coeff.', 'cuSPARSE\nreuse', 'AtBSR\nreuse', 'Mixed tile\nreuse']
    colors = ['#8a8a8a', '#a9bfd5', '#4878a8', '#e79b38', '#b44343', '#5b9864']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, case in zip(axes, ('coherent-r5', 'coherent-r7')):
        data = [s for s in summary if s['case'] == case]
        means = np.array([r['median_total_s'] for r in data])
        err = np.array([[r['median_total_s']-r['min_total_s'] for r in data], [r['max_total_s']-r['median_total_s'] for r in data]])
        ax.bar(range(len(data)), means, color=colors, yerr=err, capsize=3)
        ax.set_xticks(range(len(data)), labels, fontsize=8)
        ax.set_ylabel('Application compute time (s)')
        ax.set_title(f"{data[0]['dofs']:,} DoFs; {data[0]['steps']} time steps")
        for i, value in enumerate(means): ax.text(i, value*0.5, f'{value:.2f}', ha='center', va='center', color='white', fontsize=9)
        ax.set_ylim(0, max(r['max_total_s'] for r in data)*1.08)
    fig.suptitle('MFEM nonlinear heat conduction: four trials, median with min/max range')
    fig.tight_layout()
    fig.savefig(output/'application-times.png', dpi=200)
    fig.savefig(output/'application-times.pdf')
    plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!='summary'}, indent=2))


if __name__ == '__main__':
    main()
