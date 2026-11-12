#!/usr/bin/env python3
"""Measure actual tuned Plan selection against default AtBSR and cuSPARSE.

The manifest is randomized before any observations. A foreign GPU process
invalidates the current measurement and causes a retry after an idle grace.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

from benchmark_dsm_atbsr_full import apps, sha, write_json


def gpu_state(index):
    line = subprocess.check_output(['nvidia-smi', '-i', str(index), '--query-gpu=utilization.gpu,memory.free', '--format=csv,noheader,nounits'], text=True).strip()
    util, free = (int(x.strip()) for x in line.split(','))
    return {'utilization_percent': util, 'free_memory_mib': free}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--matrix-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-index', type=int, required=True)
    p.add_argument('--allow-pid', type=int, action='append', default=[], help='Explicitly authorized resident GPU process; results are shared-device measurements')
    p.add_argument('--seed', type=int, default=20261028)
    p.add_argument('--trials', type=int, default=9)
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    out = a.output
    for d in (out, out/'raw', out/'status'):
        d.mkdir(parents=True, exist_ok=True)
    configs = out/'configs.csv'
    expected_config = 'atbsr,-1,-1,8,128,1,1024\nauto_tuned,-1,-1,8,128,1,1024\n'
    if configs.exists() and configs.read_text() != expected_config:
        raise RuntimeError('configuration mismatch')
    configs.write_text(expected_config)
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu_index), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    rows = list(csv.DictReader(a.index.open()))
    if len({r['dataset'] for r in rows}) != len(rows) or any(Path(r['dataset']).name != r['dataset'] for r in rows):
        raise ValueError('duplicate or invalid dataset names')
    random.Random(a.seed).shuffle(rows)
    meta = {'index_sha256': sha(a.index), 'exe_sha256': sha(a.exe), 'config_sha256': sha(configs),
            'runner_sha256': sha(Path(__file__)), 'gpu_uuid': uuid, 'seed': a.seed, 'trials': a.trials, 'timeout_s': a.timeout, 'allowed_gpu_pids': sorted(a.allow_pid),
            'method': 'Actual Auto+tune selected before independent interleaved warm Graph timings; compare default AtBSR and preprocessed cuSPARSE CSR_ALG1. Each candidate checked with two nonuniform vectors and a graph against CPU long-double reference. Preparation excluded.'}
    if (out/'metadata.json').exists():
        if json.loads((out/'metadata.json').read_text()) != meta:
            raise RuntimeError('metadata mismatch; use a fresh output directory')
    else:
        write_json(out/'metadata.json', meta)
        write_json(out/'manifest.json', rows)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(a.gpu_index)
    count = 0
    grace = True
    for index, row in enumerate(rows):
        name = row['dataset']
        status = out/'status'/f'{name}.json'
        if status.exists():
            continue
        matrix = a.matrix_root/name/(name+'.mtx')
        if row['status'] != 'ok' or not matrix.is_file():
            write_json(status, {'dataset': name, 'status': 'missing', 'path': str(matrix)})
            continue
        idle_since = None
        while True:
            foreign = [x for x in apps(uuid) if x['pid'] not in a.allow_pid]
            if foreign:
                grace = True
                idle_since = None
                print('WAIT GPU busy', foreign, flush=True)
                time.sleep(15)
                continue
            state = gpu_state(a.gpu_index) if a.allow_pid else None
            if state and (state['utilization_percent'] or state['free_memory_mib'] < 1024):
                idle_since = None
                grace = True
                print('WAIT shared GPU utilization/memory', state, flush=True)
                time.sleep(5)
                continue
            if not grace:
                break
            if idle_since is None:
                idle_since = time.monotonic()
            if time.monotonic() - idle_since >= 30:
                grace = False
                break
            time.sleep(5)
        result = {'dataset': name, 'matrix_sha256': sha(matrix), 'index_nnz': int(row['nnz']), 'gpu_before': gpu_state(a.gpu_index), 'allowed_gpu_pids': sorted(a.allow_pid)}
        tmp = out/'raw'/f'{name}.csv.tmp'
        log = out/'raw'/f'{name}.log.tmp'
        start = time.monotonic()
        contested = False
        timed_out = False
        with tmp.open('w') as f, log.open('w') as e:
            proc = subprocess.Popen([str(a.exe.resolve()), str(matrix.resolve()), str(configs.resolve()), str(a.seed), str(a.trials)], stdout=f, stderr=e, env=env, start_new_session=True)
            while proc.poll() is None:
                time.sleep(.5)
                try:
                    foreign = [x for x in apps(uuid) if x['pid'] != proc.pid and x['pid'] not in a.allow_pid]
                except Exception:
                    foreign = [{'pid': -1, 'name': 'GPU monitor error'}]
                if foreign or time.monotonic()-start > a.timeout:
                    contested = bool(foreign)
                    timed_out = not contested
                    os.killpg(proc.pid, signal.SIGTERM)
                    break
            try:
                rc = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                rc = proc.wait()
        if contested:
            print('RETRY due to external GPU use', name, foreign, flush=True)
            tmp.unlink(missing_ok=True)
            log.unlink(missing_ok=True)
            return 75  # wrapper restarts; discarded case remains pending
        os.replace(tmp, out/'raw'/f'{name}.csv')
        os.replace(log, out/'raw'/f'{name}.log')
        result.update(status='timeout' if timed_out else ('ok' if rc == 0 else 'error'), returncode=rc, elapsed_s=round(time.monotonic()-start, 3), gpu_after=gpu_state(a.gpu_index))
        if rc == 0:
            with (out/'raw'/f'{name}.csv').open() as f:
                records = list(csv.DictReader(f))
            actual = {(r['backend'], int(r['trial'])) for r in records}
            expected = {(backend, trial) for backend in ('atbsr', 'auto_tuned', 'cusparse') for trial in range(a.trials)}
            if len(records) != len(expected) or actual != expected or any(not 0 < float(r['us']) < float('inf') for r in records):
                result['status'] = 'invalid_output'
        write_json(status, result)
        count += 1
        if count % 10 == 0 or result['status'] != 'ok':
            print(f'{index+1}/{len(rows)} {name}: {result["status"]}, new={count}', flush=True)
        if a.limit and count >= a.limit:
            break
    print('DONE new cases', count, flush=True)


if __name__ == '__main__':
    main()
