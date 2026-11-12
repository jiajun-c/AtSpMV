#!/usr/bin/env python3
"""Measure cached-tile CTA sizes in counterbalanced order."""
import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--factors', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=90)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--include-1024', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    for name in json.loads(a.manifest.read_text())['names']:
        factor = a.factors / (name + '-L.mtx')
        if not factor.exists():
            (a.output / (name + '-skip.json')).write_text(json.dumps({'status': 'factor_missing'}) + '\n')
            continue
        for order in ('forward', 'reverse'):
            target = a.output / f'{name}-{order}.json'
            if a.resume and target.exists():
                try:
                    if json.loads(target.read_text()).get('returncode') == 0:
                        continue
                except (OSError, ValueError):
                    pass
            env = dict(os.environ, ATSPTRSV_CACHED_ONLY='1')
            if a.include_1024: env['ATSPTRSV_THREADS_512_1024'] = '1'
            else: env.pop('ATSPTRSV_THREADS_512_1024', None)
            if order == 'reverse': env['ATSPTRSV_REVERSE_THREADS'] = '1'
            else: env.pop('ATSPTRSV_REVERSE_THREADS', None)
            try:
                r = subprocess.run([str(a.exe), str(factor), '--factor'], capture_output=True,
                                   text=True, timeout=a.timeout, env=env)
                result = {'name': name, 'order': order, 'returncode': r.returncode,
                          'stdout': r.stdout, 'stderr': r.stderr}
            except subprocess.TimeoutExpired as e:
                result = {'name': name, 'order': order, 'timeout_seconds': a.timeout,
                          'stdout': (e.stdout or b'').decode(errors='replace'),
                          'stderr': (e.stderr or b'').decode(errors='replace')}
            target.write_text(json.dumps(result, indent=2) + '\n')
            print(name, order, result.get('returncode', 'timeout'), flush=True)


if __name__ == '__main__':
    main()
