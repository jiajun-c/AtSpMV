#!/usr/bin/env python3
"""Audit the independent cuSPARSE/AtBSR application comparison and ablations."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np


def audit(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    if hashlib.sha256(Path(manifest['exe']).read_bytes()).hexdigest() != manifest['executable_sha256']:
        raise ValueError('application executable changed')
    if not manifest['persist_x'] or not manifest['gpu_assembly']:
        raise ValueError('missing common application optimization')
    backends = manifest['backends'].split(',')
    trials = []
    for rnd in range(manifest['rounds']):
        reference = np.fromfile(folder/f'cusparse-{rnd}.bin', dtype=np.float64)
        baseline = None
        for backend in backends:
            rec = json.loads((folder/f'{backend}-{rnd}.json').read_text())
            r = rec.get('result') or {}
            cmd = rec['command']
            expected = {'-b': backend, '-r': str(manifest['refine']), '-o': str(manifest['order']),
                        '-a': str(manifest['alpha']), '-s': '21', '-dt': str(manifest['dt']),
                        '-tf': str(manifest['steps']*manifest['dt'])}
            if any(cmd[cmd.index(k)+1] != v for k, v in expected.items()):
                raise ValueError('unexpected application configuration')
            if any(k not in cmd for k in ('-persist-x', '-fa')) or any(k in cmd for k in ('-rebuild', '-rebuild-t', '-dump-t')):
                raise ValueError('unequal optimization settings or diagnostic dump in timing')
            if rec['returncode'] or r['steps'] != manifest['steps']:
                raise ValueError('failed or incomplete process')
            if backend == 'cusparse':
                baseline = r
            if baseline is None:
                raise ValueError('list cuSPARSE first in the experiment manifest')
            if r['cg_iterations'] != baseline['cg_iterations'] or r['spmv_calls'] != baseline['spmv_calls']:
                raise ValueError('different iteration or matrix multiply counts')
            if r['mass_relative_error'] > 1e-6 or r['final_energy'] > r['initial_energy']*(1+1e-6):
                raise ValueError('physical invariant failed')
            field = np.fromfile(folder/f'{backend}-{rnd}.bin', dtype=np.float64)
            if len(field) != r['dofs'] or field.shape != reference.shape or not np.isfinite(field).all():
                raise ValueError('invalid full field')
            error = float(np.max(np.abs(field-reference)))/max(1., float(np.max(np.abs(reference))))
            if error > 1e-7:
                raise ValueError('incorrect full field')
            for k in ('total_s', 'linear_s', 'evolution_s'):
                if not math.isfinite(r[k]) or r[k] <= 0:
                    raise ValueError('invalid timing')
            trials.append({'case': folder.name, 'round': rnd, 'backend': backend,
                           **{k: r[k] for k in ('dofs', 'nnz', 'steps', 'cg_iterations', 'spmv_calls',
                                                'total_s', 'linear_s', 'evolution_s', 'plan_s')},
                           'process_wall_s': rec['process_wall_s'], 'field_relative_error': error})
    return trials


def summarize(trials, backend):
    rows = [r for r in trials if r['backend'] == backend]
    ref = {r['round']: r for r in trials if r['backend'] == 'cusparse'}
    result = {'trials': len(rows)}
    for key in ('total_s', 'linear_s', 'evolution_s', 'plan_s', 'process_wall_s'):
        result['median_'+key] = statistics.median(r[key] for r in rows)
        logs = np.array([math.log(ref[r['round']][key]/r[key]) for r in rows])
        result['geomean_'+key+'_speedup'] = float(np.exp(logs.mean()))
        # Conditional interval over these paired trials, not across matrices.
        samples = np.random.default_rng(20261104).integers(0, len(rows), (100000, len(rows)))
        result[key+'_paired_bootstrap_95'] = np.exp(np.quantile(logs[samples].mean(axis=1), [.025, .975])).tolist()
    result['application_wins'] = sum(r['total_s'] < ref[r['round']]['total_s'] for r in rows)
    result['max_field_relative_error'] = max(r['field_relative_error'] for r in rows)
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/applications/mfem-heat'))
    a = p.parse_args()
    output = a.root/'optimized-report'; output.mkdir(exist_ok=True)
    cases = ['optimize-compressed-persist-v2-r7', 'optimized-confirmation-r7']
    report = {'cases': {}, 'scope': 'H100 PCIe; FP64 nonlinear heat; 1313281 DoFs; 20981761 CSR entries; 20 steps. Both backends use value/structure reuse and persisting X. Bootstrap intervals describe repeated trials on this case only.'}
    all_trials = []
    for case in cases:
        trials = audit(a.root/case); all_trials.extend(trials)
        report['cases'][case] = {b: summarize(trials, b) for b in dict.fromkeys(r['backend'] for r in trials)}
    kernels = []
    for persist in (0, 1):
        for rnd in range(3):
            for r in csv.DictReader((a.root/'optimized-kernels'/f'persist-{persist}-{rnd}.csv').open()):
                if r['candidate'] in ('cusparse', 'atbsr-w4-d2', 'compressed-w4'):
                    kernels.append({'persist_x': persist, 'round': rnd, 'candidate': r['candidate'], 'median_us': float(r['median_us'])})
    profile = []
    for case in ('atbsr', 'atbsr-persist', 'compressed-persist'):
        lines=(a.root/'optimized-profile'/(case+'.csv')).read_text().splitlines()
        lines=lines[next(i for i, line in enumerate(lines) if line.startswith('"ID"')):]
        metrics={r['Metric Name']: float(r['Metric Value']) for r in csv.DictReader(lines)}
        profile.append({'case': case, 'dram_bytes_read': metrics['dram__bytes_read.sum'],
                        'aggregate_l2_hit_percent': metrics['lts__t_sector_hit_rate.pct']})
    report['profile'] = profile
    write_csv(output/'trials.csv', all_trials); write_csv(output/'kernels.csv', kernels); write_csv(output/'profile.csv', profile)
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, name in enumerate(('cusparse', 'atbsr-w4-d2', 'compressed-w4')):
        values = [statistics.median(r['median_us'] for r in kernels if r['candidate']==name and r['persist_x']==j) for j in (0,1)]
        axes[0].bar(np.arange(2)+(i-1)*.25, values, width=.25, label=name)
    axes[0].set_xticks([0,1], ['Default cache', 'Persisting X (all backends)'])
    axes[0].set_ylabel('SpMV kernel time (us)'); axes[0].legend(fontsize=8)
    confirm=[r for r in all_trials if r['case']==cases[1]]
    for i, backend in enumerate(('cusparse', 'atbsr-compressed')):
        values=[r['total_s'] for r in confirm if r['backend']==backend]; med=statistics.median(values)
        axes[1].bar(i, med, yerr=[[med-min(values)], [max(values)-med]], capsize=4)
        axes[1].text(i, med*.5, f'{med:.3f}', ha='center', color='white')
    axes[1].set_xticks([0,1], ['cuSPARSE', 'Compressed AtBSR'])
    axes[1].set_ylabel('Application compute time (s)')
    axes[1].set_title('Independent confirmation: eight trials, median/range')
    fig.tight_layout(); fig.savefig(output/'comparison.png', dpi=200); fig.savefig(output/'comparison.pdf'); plt.close(fig)
    print(json.dumps(report['cases'][cases[1]], indent=2))


if __name__ == '__main__':
    main()
