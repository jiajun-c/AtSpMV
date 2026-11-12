#!/usr/bin/env python3
"""Audit that Graph comparison matrices match the prior v2 test inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path


def audit(reference, runs):
    seen, states, problems = set(), Counter(), []
    for run in runs:
        manifest = json.loads((run/'manifest.json').read_text())
        for item in manifest:
            name = item['dataset']
            if name in seen: raise ValueError(f'duplicate dataset: {name}')
            seen.add(name)
            status = run/'status'/f'{name}.json'
            if not status.exists():
                states['pending'] += 1
                continue
            state = json.loads(status.read_text())
            result = state.get('status','missing_status')
            states[result] += 1
            if result != 'ok': continue
            ref_path = reference/'status'/f'{name}.json'
            if not ref_path.exists():
                problems.append({'dataset':name,'reason':'missing reference status'})
                continue
            ref = json.loads(ref_path.read_text())
            if ref.get('status') != 'ok' or not state.get('matrix_sha256') or state['matrix_sha256'] != ref.get('matrix_sha256'):
                problems.append({'dataset':name,'reason':'input hash or reference status mismatch'})
    return {'manifest_cases':len(seen),'status_counts':dict(states),'problems':problems,
            'method':'Compare recorded input SHA-256 and successful status against the original v2 run. '
                     'This checks matrix identity, not numerical correctness or measurement interference.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--run',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=audit(a.reference,a.run)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result['problems']: raise SystemExit(1)


if __name__ == '__main__': main()
