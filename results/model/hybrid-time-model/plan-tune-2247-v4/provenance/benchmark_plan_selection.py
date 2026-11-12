#!/usr/bin/env python3
"""Measure actual tuned Plan selection against default AtBSR and cuSPARSE.

The manifest is randomized before any observations. A foreign GPU process
invalidates the current measurement and causes a retry after an idle grace.
"""
import argparse
import csv
import hashlib
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


def recently_active_resident(nvml, device, allowed, seconds=3):
    """Use per-process samples so our last kernel is not mistaken for a resident."""
    try:
        samples=nvml.nvmlDeviceGetProcessUtilization(device,int((time.time()-seconds)*1000000))
    except nvml.NVMLError_NotFound:
        return []
    return [{'pid':x.pid,'sm':x.smUtil,'mem':x.memUtil,'timestamp':x.timeStamp}
            for x in samples if x.pid in allowed and (x.smUtil or x.memUtil)]


def shared_gpu_requires_wait(state,resident,grace,monitor_resident):
    return bool(resident or (state and state['free_memory_mib']<1024) or
                (state and state['utilization_percent'] and (grace or not monitor_resident)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--matrix-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-index', type=int, required=True)
    p.add_argument('--allow-pid', type=int, action='append', default=[], help='Explicitly authorized resident GPU process; results are shared-device measurements')
    p.add_argument('--monitor-resident', action='store_true', help='Discard cases when NVML reports SM or memory activity for an allowed resident process')
    p.add_argument('--selection', choices=('auto_tuned','auto_model'), default='auto_tuned')
    p.add_argument('--counterbalance', action='store_true', help='Repeat each case with reversed Plan construction order')
    p.add_argument('--extended-baselines', action='store_true', default=True, help='Also time the fixed AtBSR and CSR baselines used by the prior full sweep')
    p.add_argument('--vector-candidates', action='store_true', help='Also measure six CSR vector widths')
    p.add_argument('--seed', type=int, default=20261028)
    p.add_argument('--trials', type=int, default=9)
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    if a.vector_candidates and not a.extended_baselines:
        p.error('--vector-candidates requires --extended-baselines')
    out = a.output
    for d in (out, out/'raw', out/'status'):
        d.mkdir(parents=True, exist_ok=True)
    configs = out/'configs.csv'
    lines=['atbsr,-1,-1,8,128,1,1024']
    if a.extended_baselines:
        lines.extend(['atbsr_m8,3,0,8,128,1,1024','atbsr_m1_balanced,0,1,8,128,1,1024',
                      'csr_dsm_fixed,-1,-1,8,256,1,1024','csr_two_stage_fixed,-1,-1,8,256,1,1024'])
    if a.vector_candidates:
        lines.extend(f'csr_vec{width},-1,-1,{width},128,1,1024' for width in (1,2,4,8,16,32))
    lines.append(f'{a.selection},-1,-1,8,128,1,1024')
    expected_config = '\n'.join(lines)+'\n'
    if configs.exists() and configs.read_text() != expected_config:
        raise RuntimeError('configuration mismatch')
    configs.write_text(expected_config)
    reverse_configs = out/'configs-reversed.csv'
    if a.counterbalance:
        reverse = '\n'.join(reversed(lines))+'\n'
        if reverse_configs.exists() and reverse_configs.read_text() != reverse:
            raise RuntimeError('reversed configuration mismatch')
        reverse_configs.write_text(reverse)
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu_index), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    rows = list(csv.DictReader(a.index.open()))
    if len({r['dataset'] for r in rows}) != len(rows) or any(Path(r['dataset']).name != r['dataset'] for r in rows):
        raise ValueError('duplicate or invalid dataset names')
    random.Random(a.seed).shuffle(rows)
    meta = {'index_sha256': sha(a.index), 'exe_sha256': sha(a.exe), 'config_sha256': sha(configs),
            'runner_sha256': sha(Path(__file__)), 'gpu_uuid': uuid, 'seed': a.seed, 'trials': a.trials, 'timeout_s': a.timeout, 'allowed_gpu_pids': sorted(a.allow_pid), 'monitor_resident': a.monitor_resident, 'selection':a.selection,
            'counterbalance':a.counterbalance, 'extended_baselines':a.extended_baselines, 'vector_candidates':a.vector_candidates, 'reverse_config_sha256':sha(reverse_configs) if a.counterbalance else None,
            'idle_policy':'30s GPU-wide idle at startup/after interference; otherwise use per-process resident activity, foreign process detection and memory reserve. Polling cannot rule out short or non-compute interference.',
            'method': 'Actual Plan selection before independent interleaved warm Graph timings; compare default AtBSR and preprocessed cuSPARSE CSR_ALG1. Each candidate checked with two nonuniform vectors and a graph against CPU long-double reference. Preparation excluded.'}
    if (out/'metadata.json').exists():
        if json.loads((out/'metadata.json').read_text()) != meta:
            raise RuntimeError('metadata mismatch; use a fresh output directory')
    else:
        write_json(out/'metadata.json', meta)
        write_json(out/'manifest.json', rows)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(a.gpu_index)
    if a.monitor_resident:
        import pynvml
        pynvml.nvmlInit()
        nvml_device = pynvml.nvmlDeviceGetHandleByIndex(a.gpu_index)
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
            # GPU-wide utilization can include this runner's just-completed
            # kernels. Once the initial grace has passed, use per-process
            # samples for authorized residents; continue to reject all other
            # compute processes and require a memory reserve.
            try:
                resident = recently_active_resident(pynvml,nvml_device,a.allow_pid) if a.monitor_resident else []
            except Exception as e:
                resident = [{'pid':-1,'name':'NVML monitor error: '+str(e)}]
            if shared_gpu_requires_wait(state,resident,grace,a.monitor_resident):
                idle_since = None
                grace = True
                print('WAIT shared GPU activity/memory', state, resident, flush=True)
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
        last_sample = int(time.time()*1000000)
        def active_resident():
            nonlocal last_sample
            if not a.monitor_resident:
                return []
            try:
                samples = pynvml.nvmlDeviceGetProcessUtilization(nvml_device, last_sample)
            except pynvml.NVMLError_NotFound:
                return []
            if samples:
                last_sample = max(x.timeStamp for x in samples)
            return [{'pid': x.pid, 'sm': x.smUtil, 'mem': x.memUtil, 'timestamp': x.timeStamp} for x in samples if x.pid in a.allow_pid and (x.smUtil or x.memUtil)]
        orders=[('baseline_first',configs)]
        if a.counterbalance:
            orders.append(('selection_first',reverse_configs))
            if int(hashlib.sha256(f'{a.seed}:{name}'.encode()).hexdigest()[:2],16)%2:orders.reverse()
        collected=[];rc=0
        with log.open('w') as e:
            for ordinal,(label,config) in enumerate(orders):
                e.write(f'BENCH_ORDER,{label}\n');e.flush()
                with tmp.open('w') as f:
                    proc = subprocess.Popen([str(a.exe.resolve()), str(matrix.resolve()), str(config.resolve()), str(a.seed+ordinal), str(a.trials)], stdout=f, stderr=e, env=env, start_new_session=True)
                    while proc.poll() is None:
                        time.sleep(.5)
                        try:
                            foreign = [x for x in apps(uuid) if x['pid'] != proc.pid and x['pid'] not in a.allow_pid] + active_resident()
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
                if contested or rc!=0:break
                with tmp.open() as f:
                    part=list(csv.DictReader(f))
                for record in part:record['order']=label
                collected.extend(part)
        if not contested and a.monitor_resident:
            foreign = active_resident()
            contested = bool(foreign)
        if contested:
            print('RETRY due to external GPU use', name, foreign, flush=True)
            tmp.unlink(missing_ok=True)
            log.unlink(missing_ok=True)
            return 75  # wrapper restarts; discarded case remains pending
        if rc==0:
            with tmp.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['backend','config','trial','us','order'])
                writer.writeheader();writer.writerows(collected)
        os.replace(tmp, out/'raw'/f'{name}.csv')
        os.replace(log, out/'raw'/f'{name}.log')
        result.update(status='timeout' if timed_out else ('ok' if rc == 0 else 'error'), returncode=rc, elapsed_s=round(time.monotonic()-start, 3), gpu_after=gpu_state(a.gpu_index))
        if rc == 0:
            with (out/'raw'/f'{name}.csv').open() as f:
                records = list(csv.DictReader(f))
            actual = {(r['backend'], int(r['trial']),r['order']) for r in records}
            baseline_names=['atbsr','cusparse']+(['atbsr_m8','atbsr_m1_balanced','csr_dsm_fixed','csr_two_stage_fixed'] if a.extended_baselines else [])
            if a.vector_candidates:baseline_names += [f'csr_vec{x}' for x in (1,2,4,8,16,32)]
            expected = {(backend, trial,label) for label,_ in orders for backend in baseline_names+[a.selection] for trial in range(a.trials)}
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
    raise SystemExit(main())
