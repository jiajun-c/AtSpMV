"""Run after the full application sweep, on the same otherwise idle GPU."""
import hashlib
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import statistics

ROOT = Path(__file__).resolve().parent
APP = '/tmp/atspmv-mfem-compressed-v3'
BENCH = '/tmp/atspmv-mfem-apps/apps/mfem/packed_spmv_bench'
MESH = '/tmp/atspmv-mfem-v4.8/data/star.mesh'
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='3', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
           ATSPMV_TUNE_PREFETCH='2', ATSPMV_TUNE_WARPS='4', ATSPMV_TUNE_CAP='8', ATSPMV_TUNE_CTA_REDUCE='0')


def run(cmd, log, err=None):
    with log.open('w') as out:
        if err:
            with err.open('w') as errors:
                subprocess.run(cmd, env=ENV, stdout=out, stderr=errors, check=True, timeout=180)
        else:
            subprocess.run(cmd, env=ENV, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=180)


def app_command(refine, backend):
    return [APP, '-m', MESH, '-r', str(refine), '-o', '2', '-s', '21', '-dt', '.01',
            '-tf', '.01', '-a', '.01', '-fa', '-b', backend, '-sol', '/tmp/atspmv-ablation-diagnostic.bin']


def collect():
    if not (ROOT/'completed.json').exists():
        raise RuntimeError('application sweep has not completed; avoid concurrent GPU measurements')
    for refine in (5, 7):
        folder = ROOT/f'kernels-r{refine}'; folder.mkdir(exist_ok=True)
        matrix = f'/tmp/atspmv-ablation-r{refine}.csr'
        export = app_command(refine, 'cusparse')+['-dump-t', matrix]
        run(export, folder/'export.log')
        commands = [export]
        for rnd in range(3):
            for persist in ((0, 1) if rnd % 2 == 0 else (1, 0)):
                cmd = [BENCH, matrix]+(['--persist-x'] if persist else [])
                run(cmd, folder/f'persist-{persist}-{rnd}.csv', folder/f'persist-{persist}-{rnd}.log')
                commands.append(cmd)
        for backend in ('atbsr', 'atbsr-compressed'):
            kernel = '.*compressedTileSpmv.*' if backend.endswith('compressed') else '.*tunedSpmv.*'
            profile = folder/(backend+'-profile.csv')
            cmd = ['/usr/local/cuda-12.4/bin/ncu', '--cache-control', 'none', '--clock-control', 'none',
                   '--kernel-name-base', 'demangled', '--kernel-name', 'regex:'+kernel,
                   '--launch-skip', '100', '--launch-count', '1', '--metrics',
                   'dram__bytes_read.sum,lts__t_sector_hit_rate.pct,smsp__inst_executed.sum',
                   '--csv', '--log-file', str(profile), *app_command(refine, backend), '-persist-x']
            run(cmd, folder/(backend+'-profile.log')); commands.append(cmd)
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        (folder/'manifest.json').write_text(json.dumps({'commands': commands, 'env': {k: ENV[k] for k in ENV if k.startswith('ATSPMV_') or k=='CUDA_VISIBLE_DEVICES'},
            'matrix_sha256': sha(matrix), 'application_sha256': sha(APP), 'benchmark_sha256': sha(BENCH)}, indent=2)+'\n')
        print(f'kernel replay and profile r{refine} complete', flush=True)


def summarize():
    summary = {'kernels': [], 'profiles': []}
    for refine in (5, 7):
        folder = ROOT/f'kernels-r{refine}'
        for persist in (0, 1):
            rows = []
            for rnd in range(3):
                rows.extend(csv.DictReader((folder/f'persist-{persist}-{rnd}.csv').open()))
            values = {b: statistics.median(float(r['median_us']) for r in rows if r['candidate'] == b)
                      for b in ('cusparse', 'atbsr-w4-d2', 'compressed-w4')}
            summary['kernels'].append({'refine': refine, 'persist_x': persist, **values,
                'compression_speedup': values['atbsr-w4-d2']/values['compressed-w4']})
        for backend in ('atbsr', 'atbsr-compressed'):
            lines = (folder/(backend+'-profile.csv')).read_text().splitlines()
            start = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
            metrics = {r['Metric Name']: float(r['Metric Value']) for r in csv.DictReader(lines[start:])}
            summary['profiles'].append({'refine': refine, 'backend': backend, **metrics})
    (ROOT/'report/diagnostics.json').write_text(json.dumps(summary, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, refine in zip(axes, (5, 7)):
        rows = [r for r in summary['kernels'] if r['refine'] == refine]
        for offset, key, label in [(-.18, 'atbsr-w4-d2', 'AtBSR'), (.18, 'compressed-w4', 'Compressed AtBSR')]:
            values = [next(r[key] for r in rows if r['persist_x'] == p) for p in (0, 1)]
            ax.bar(np.arange(2)+offset, values, width=.36, label=label)
        ax.set_xticks([0, 1], ['Default cache', 'Persisting X'])
        ax.set_ylabel('SpMV kernel time (us)')
        ax.set_title('82,561 DoFs' if refine == 5 else '1,313,281 DoFs')
        ax.set_ylim(0, max(r[k] for r in rows for k in ('atbsr-w4-d2', 'compressed-w4'))*1.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT/'report/kernel-ablation.png', dpi=200)
    fig.savefig(ROOT/'report/kernel-ablation.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analyze-only', action='store_true')
    if not parser.parse_args().analyze_only:
        collect()
    summarize()
