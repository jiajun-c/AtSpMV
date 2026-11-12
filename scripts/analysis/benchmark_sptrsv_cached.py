#!/usr/bin/env python3
"""Run cached-tile/cuSPARSE and patched Recblock on the same factors."""
import argparse
import json
import os
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--factors', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--ours', type=Path, required=True)
p.add_argument('--recblock', type=Path, required=True)
p.add_argument('--timeout', type=int, default=40)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
for name in json.loads(a.manifest.read_text())['names']:
    factor = a.factors / (name + '-L.mtx')
    if not factor.exists():
        (a.output / (name + '-skipped.json')).write_text(json.dumps({'name': name, 'reason': 'factorization failed or factor missing'}) + '\n')
        print(name, 'skipped: no factor', flush=True)
        continue
    cmds = {'ours': [str(a.ours), str(factor), '--factor'],
            'recblock': [str(a.recblock), '-d', '0', '-rhs', '1', '-lv', '-1', '-forward', '-mtx', str(factor)]}
    for which in ('ours', 'recblock'):
        env = dict(os.environ)
        if which == 'ours': env['ATSPTRSV_CACHED_ONLY'] = '1'
        else: env['LD_LIBRARY_PATH'] = '/usr/local/cuda-11.8/lib64'
        try:
            r = subprocess.run(cmds[which], capture_output=True, text=True, timeout=a.timeout, env=env)
            obj = {'name': name, 'implementation': which, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
        except subprocess.TimeoutExpired as e:
            obj = {'name': name, 'implementation': which, 'timeout_seconds': a.timeout,
                   'stdout': (e.stdout or b'').decode(errors='replace'), 'stderr': (e.stderr or b'').decode(errors='replace')}
        (a.output / (name + '-' + which + '.json')).write_text(json.dumps(obj, indent=2) + '\n')
        print(name, which, obj.get('returncode', 'timeout'), flush=True)
