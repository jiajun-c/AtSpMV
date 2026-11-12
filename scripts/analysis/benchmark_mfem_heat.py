#!/usr/bin/env python3
"""Run and validate complete MFEM heat applications in counterbalanced order."""
import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--mesh', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--refine', type=int, default=5)
    p.add_argument('--order', type=int, default=2)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--dt', type=float, default=0.01)
    p.add_argument('--alpha', type=float, default=0)
    p.add_argument('--backends', default='cusparse,atbsr,mixed')
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--timeout', type=float, default=900)
    p.add_argument('--rebuild-values', action='store_true')
    p.add_argument('--prefetch', type=int, choices=(0,1,2,4,8), default=0)
    p.add_argument('--warps', type=int, choices=(2,4,8), default=4)
    p.add_argument('--gpu-assembly', action='store_true')
    p.add_argument('--rebuild-implicit', action='store_true')
    p.add_argument('--persist-x', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    backends = a.backends.split(',')
    if 'cusparse' not in backends:
        p.error('include cusparse for validation')
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    manifest['executable_sha256'] = hashlib.sha256(a.exe.read_bytes()).hexdigest()
    manifest_path = a.output/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError('output directory already contains a different experiment; use a new directory')
    write_json(manifest_path, manifest)
    records = {}
    for rnd in range(a.rounds):
        for backend in (backends if rnd % 2 == 0 else backends[::-1]):
            stem = a.output / f'{backend}-{rnd}'
            dest = stem.with_suffix('.json')
            solution = stem.with_suffix('.bin')
            if dest.exists() and solution.exists():
                try:
                    record = json.loads(dest.read_text())
                except (OSError, ValueError):
                    record = {}
                expected = record.get('result') or {}
                if record.get('returncode') == 0 and solution.stat().st_size == expected.get('dofs', -1)*8:
                    records[rnd, backend] = record
                    continue
            local_output = tempfile.TemporaryDirectory(prefix='atspmv-heat-field-')
            local_solution = Path(local_output.name) / 'temperature.bin'
            app_backend = backend
            reference_rebuild = backend in ('mfem-rebuild', 'cusparse-rebuild', 'atbsr-rebuild')
            host_coefficients = backend in ('mfem-hostcoeff', 'cusparse-hostcoeff', 'atbsr-hostcoeff')
            if reference_rebuild:
                app_backend = backend.removesuffix('-rebuild')
            if host_coefficients:
                app_backend = backend.removesuffix('-hostcoeff')
            cmd = [str(a.exe.resolve()), '-m', str(a.mesh.resolve()), '-r', str(a.refine),
                   '-o', str(a.order), '-s', '21', '-dt', str(a.dt), '-tf', str(a.steps*a.dt),
                   '-a', str(a.alpha), '-b', app_backend, '-sol', str(local_solution)]
            if a.rebuild_values or reference_rebuild:
                cmd.append('-rebuild')
            if a.gpu_assembly:
                cmd.append('-fa')
            if a.persist_x:
                cmd.append('-persist-x')
            if a.rebuild_implicit or reference_rebuild:
                cmd.append('-rebuild-t')
            if reference_rebuild or host_coefficients:
                cmd.append('-host-coeff')
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '1'
            env['OPENBLAS_NUM_THREADS'] = '1'
            env['ATSPMV_TUNE_PREFETCH'] = str(a.prefetch)
            env['ATSPMV_TUNE_WARPS'] = str(a.warps)
            env['ATSPMV_TUNE_CAP'] = '8'
            env['ATSPMV_TUNE_CTA_REDUCE'] = '0'
            start = time.perf_counter()
            run = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=a.timeout)
            wall = time.perf_counter() - start
            if local_solution.exists():
                shutil.copyfile(local_solution, solution)
            local_output.cleanup()
            stem.with_suffix('.log').write_text(run.stdout + '\nSTDERR\n' + run.stderr)
            found = [s for s in run.stdout.splitlines() if s.startswith('APP_RESULT ')]
            record = {'backend': backend, 'round': rnd, 'command': cmd, 'returncode': run.returncode, 'process_wall_s': wall,
                      'result': json.loads(found[-1][11:]) if found else None}
            write_json(dest, record)
            print(rnd, backend, run.returncode, record['result'], flush=True)
            if run.returncode or record['result'] is None:
                raise RuntimeError(f'application failed: {stem}')
            if record['result']['steps'] != a.steps:
                raise RuntimeError(f'wrong step count: {stem}')
            records[rnd, backend] = record

    comparisons = []
    for rnd in range(a.rounds):
        ref = np.fromfile(a.output / f'cusparse-{rnd}.bin', dtype=np.float64)
        if not len(ref) or not np.isfinite(ref).all():
            raise RuntimeError('nonfinite or empty reference')
        reference = records[rnd, 'cusparse']
        for backend in backends:
            got = np.fromfile(a.output / f'{backend}-{rnd}.bin', dtype=np.float64)
            if got.shape != ref.shape or not np.isfinite(got).all():
                raise RuntimeError(f'invalid field: {backend} {rnd}')
            err = float(np.max(np.abs(got-ref))) / max(1., float(np.max(np.abs(ref))))
            if err > 1e-7:
                raise RuntimeError(f'field mismatch: {backend} {rnd}: {err}')
            record = records[rnd, backend]
            comparisons.append({'round': rnd, 'backend': backend, 'field_relative_error': err,
                                'same_cg_iterations': record['result']['cg_iterations'] == reference['result']['cg_iterations'],
                                'evolution_speedup': reference['result']['evolution_s']/record['result']['evolution_s'],
                                'application_speedup': reference['result']['total_s']/record['result']['total_s'],
                                'process_wall_speedup': reference['process_wall_s']/record['process_wall_s']})
    summary = {}
    for backend in backends:
        rows = [r for r in comparisons if r['backend'] == backend]
        results = [records[r, backend]['result'] for r in range(a.rounds)]
        summary[backend] = {k: math.exp(statistics.mean(math.log(row[k]) for row in rows))
                            for k in ('evolution_speedup', 'application_speedup', 'process_wall_speedup')}
        summary[backend]['max_field_relative_error'] = max(row['field_relative_error'] for row in rows)
        summary[backend]['same_cg_iterations'] = all(row['same_cg_iterations'] for row in rows)
        summary[backend]['median_evolution_s'] = statistics.median(row['evolution_s'] for row in results)
        summary[backend]['median_application_s'] = statistics.median(row['total_s'] for row in results)
        summary[backend]['median_plan_s'] = statistics.median(row['plan_s'] for row in results)
        if all('linear_s' in row for row in results):
            for field in ('linear_s', 'parameter_s', 'implicit_setup_s'):
                summary[backend]['median_'+field] = statistics.median(row[field] for row in results)
        reference_results = [records[r, 'cusparse']['result'] for r in range(a.rounds)]
        summary[backend]['ratio_of_median_application_times'] = statistics.median(row['total_s'] for row in reference_results) / summary[backend]['median_application_s']
        summary[backend]['ratio_of_median_evolution_times'] = statistics.median(row['evolution_s'] for row in reference_results) / summary[backend]['median_evolution_s']
        if 'mfem-rebuild' in backends:
            summary[backend]['vs_mfem_rebuild_application'] = math.exp(statistics.mean(
                math.log(records[r, 'mfem-rebuild']['result']['total_s']/records[r, backend]['result']['total_s'])
                for r in range(a.rounds)))
            summary[backend]['vs_mfem_rebuild_evolution'] = math.exp(statistics.mean(
                math.log(records[r, 'mfem-rebuild']['result']['evolution_s']/records[r, backend]['result']['evolution_s'])
                for r in range(a.rounds)))
    output = {'summary': summary, 'comparisons': comparisons}
    write_json(a.output/'summary.json', output)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == '__main__':
    main()
