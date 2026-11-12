#!/usr/bin/env python3
"""Collect Nsight Compute instruction metrics for frozen cuSPARSE cases.

Run under the idle-GPU monitor; completed files are kept if a run is retried.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--exe', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--ncu', type=Path, default=Path('/usr/local/cuda-12.4/bin/ncu'))
parser.add_argument('--exact-one', action='store_true')
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
metrics = ','.join(('sm__inst_executed_pipe_tensor.sum',
                    'sm__inst_executed_pipe_tensor_op_hmma.sum',
                    'sm__sass_thread_inst_executed_op_ffma_pred_on.sum',
                    'sm__sass_thread_inst_executed_op_hfma_pred_on.sum'))
cases = [(b, 0, n) for b in (2, 8) for n in (8, 32)] + [(64, 0, 32)] if args.exact_one else [(2, 5, 32), (2, 100, 32), (8, 5, 8), (8, 100, 128)]
methods = ('csr_alg2', 'csr_alg3', 'blocked_ell')
digest = hashlib.sha256(args.exe.read_bytes()).hexdigest()
metadata = {'exe_sha256': digest, 'metrics': metrics, 'cases': [list(c) for c in cases],
            'methods': list(methods), 'launch_count': 20,
            'note': 'NCU profiles the first 20 launched kernels, including any cuSPARSE preprocessing and correctness/warmup launches; classify by kernel name, not total process counts.'}
meta = args.out/'metadata.json'
if meta.exists() and json.loads(meta.read_text()) != metadata:
    raise ValueError('metadata changed; use a new output directory')
with tempfile.NamedTemporaryFile('w', dir=args.out, delete=False) as stream:
    json.dump(metadata, stream, indent=2);stream.write('\n');tmp=stream.name
os.replace(tmp, meta)
for blocks, density, n in cases:
    for method in methods:
        stem = f'b{blocks}-d{density}-n{n}-{method}'
        status = args.out/(stem+'.json')
        if status.exists() and json.loads(status.read_text()).get('status') == 'ok':
            continue
        cmd = [str(args.ncu), '--csv', '--target-processes', 'all', '--launch-count', '20',
               '--metrics', metrics, str(args.exe.resolve()), str(blocks), str(density), str(n), method]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        for suffix, data in (('.csv', result.stdout), ('.log', result.stderr)):
            with tempfile.NamedTemporaryFile('w', dir=args.out, delete=False) as stream:
                stream.write(data); tmp = stream.name
            os.replace(tmp, args.out/(stem+suffix))
        has_metrics = 'sm__inst_executed_pipe_tensor' in result.stdout
        state = {'status': 'ok' if result.returncode == 0 and has_metrics else 'error',
                 'returncode': result.returncode, 'cmd': cmd,
                 'has_metric_records': has_metrics}
        with tempfile.NamedTemporaryFile('w', dir=args.out, delete=False) as stream:
            json.dump(state, stream, indent=2);stream.write('\n');tmp=stream.name
        os.replace(tmp, status)
        print(stem, state, flush=True)
        if result.returncode != 0 or not state['has_metric_records']:
            raise RuntimeError('Nsight Compute failed or returned no metric records; inspect logs')
print('profiling complete', flush=True)
