#!/usr/bin/env python3
"""Run a frozen, explicit SpMV factor-ablation manifest on an idle GPU."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import time

from benchmark_dsm_atbsr_full import apps, sha, write_json


def sample_key(row):
    return row['backend'] + ':' + ('auto' if row['backend'] == 'atbsr_control' else row['config'])


def analyze(output, manifest):
    rows, failures = [], []
    for item in manifest['matrices']:
        name = item['dataset']
        status_file = output/'status'/f'{name}.json'
        status = json.loads(status_file.read_text()) if status_file.exists() else {'status': 'pending'}
        if status['status'] != 'ok':
            failures.append({'dataset': name, 'status': status['status']})
            continue
        orders = []
        for order in (0, 1):
            grouped = {}
            with (output/'raw'/f'{name}-{order}.csv').open() as f:
                for row in csv.DictReader(f):
                    value = float(row['us'])
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError('invalid time')
                    grouped.setdefault(sample_key(row), []).append(value)
            if len(grouped) != len(item['configs']) + 3 or any(len(v) != manifest['trials'] for v in grouped.values()):
                raise ValueError(f'incomplete candidate/trial coverage: {name}')
            orders.append({k: statistics.median(v) for k, v in grouped.items()})
        if orders[0].keys() != orders[1].keys():
            raise ValueError('construction-order coverage differs')
        for key in orders[0]:
            rows.append({'dataset': name, 'candidate': key, 'order0_us': orders[0][key],
                         'order1_us': orders[1][key], 'geomean_us': math.sqrt(orders[0][key]*orders[1][key])})
    if rows:
        with (output/'timings.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    summary = {'requested': len(manifest['matrices']), 'successful': len(manifest['matrices'])-len(failures),
               'failures': failures, 'candidate_pairs': len(rows), 'scope': manifest['scope']}
    write_json(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--analyze-only', action='store_true')
    a = parser.parse_args(); a.output = a.output.resolve(); a.exe = a.exe.resolve()
    manifest = json.loads((a.output/'manifest.json').read_text())
    for folder in ('raw', 'status', 'configs', 'provenance'):
        (a.output/folder).mkdir(exist_ok=True)
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(manifest['gpu']), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    metadata = {'manifest_sha256': sha(a.output/'manifest.json'), 'exe_sha256': sha(a.exe),
                'runner_sha256': sha(Path(__file__)), 'gpu_uuid': uuid, 'timeout': a.timeout,
                'prefetch': 2, 'warps': 4, 'cap': 8, 'cta_reduce': 0, 'persist_x': False, 'pad_x_to_columns': 32767,
                'measurement': 'FP64 warm Graph internal events, 8 SpMVs/batch, 3 warmup rounds, randomized interleaving, two reversed construction orders; setup excluded'}
    identity = a.output/'metadata.json'
    if identity.exists() and json.loads(identity.read_text()) != metadata:
        raise ValueError('experiment identity changed')
    write_json(identity, metadata)
    if not a.analyze_only:
        env = {k: v for k, v in os.environ.items() if not k.startswith(('ATSPMV_',))}
        env.update(CUDA_VISIBLE_DEVICES=str(manifest['gpu']), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   ATSPMV_TUNE_PREFETCH='2', ATSPMV_TUNE_WARPS='4', ATSPMV_TUNE_CAP='8', ATSPMV_TUNE_CTA_REDUCE='0',
                   ATSPMV_BENCH_CUSPARSE_ALGORITHMS='1', ATSPMV_BENCH_INTERNAL_GRAPH_EVENTS='1', ATSPMV_BENCH_PAD_X='1')
        for index, item in enumerate(manifest['matrices']):
            name = item['dataset']; status_file = a.output/'status'/f'{name}.json'
            if status_file.exists() and json.loads(status_file.read_text())['status'] == 'ok':
                continue
            record = {'dataset': name, 'status': 'running', 'matrix_sha256': sha(Path(item['path'])), 'orders': []}
            write_json(status_file, record)
            for order in (0, 1):
                config = a.output/'configs'/f'{name}-{order}.csv'
                configs = item['configs'] if order == 0 else list(reversed(item['configs']))
                config.write_text(''.join(','.join(map(str, c))+'\n' for c in configs))
                while apps(uuid):
                    print('waiting for idle GPU', flush=True); time.sleep(5)
                cmd = [str(a.exe), item['path'], str(config), str(20261111+index*2+order), str(manifest['trials'])]
                start = time.monotonic(); reason = None
                with (a.output/'raw'/f'{name}-{order}.csv').open('w') as out, (a.output/'raw'/f'{name}-{order}.log').open('w') as err:
                    proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=err, start_new_session=True)
                    while proc.poll() is None:
                        foreign = [r for r in apps(uuid) if r['pid'] != proc.pid]
                        if foreign or time.monotonic()-start > a.timeout:
                            reason = 'interference' if foreign else 'timeout'
                            os.killpg(proc.pid, signal.SIGTERM)
                            try: proc.wait(timeout=3)
                            except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                            break
                        time.sleep(.5)
                record['orders'].append({'command': cmd, 'returncode': proc.returncode,
                                         'reason': reason, 'elapsed_s': time.monotonic()-start})
                if reason or proc.returncode:
                    record['status'] = reason or 'failed'
                    break
            else:
                record['status'] = 'ok'
            write_json(status_file, record)
            print(f"{index+1}/{len(manifest['matrices'])} {name}: {record['status']}", flush=True)
        shutil.copyfile(__file__, a.output/'provenance'/Path(__file__).name)
    summary = analyze(a.output, manifest)
    print(json.dumps(summary), flush=True)
    write_json(a.output/'completed.json', summary)


if __name__ == '__main__':
    main()
