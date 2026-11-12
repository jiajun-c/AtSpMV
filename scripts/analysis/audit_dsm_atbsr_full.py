#!/usr/bin/env python3
"""Audit individual sweep records and the independent targeted retests."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    meta=json.loads((root/'metadata.json').read_text());manifest=json.loads((root/'manifest.json').read_text());counts=collections.Counter();problems=[]
    expected={};
    for row in csv.reader((root/'configs.csv').open()):
        if row and not row[0].startswith('#'):expected[(row[0],'/'.join(row[1:]))]=meta['trials']
    expected[('cusparse','CSR_ALG1')]=meta['trials']
    for row in manifest:
        name=row['dataset'];status=root/'status'/f'{name}.json'
        if not status.exists():counts['pending']+=1;continue
        try:data=json.loads(status.read_text())
        except Exception as exc:problems.append({'dataset':name,'problem':str(exc)});continue
        counts[data['status']]+=1
        if data['status']!='ok':continue
        try:
            groups=collections.defaultdict(set)
            with (root/'raw'/f'{name}.csv').open() as f:
                for sample in csv.DictReader(f):
                    value=float(sample['us']);trial=int(sample['trial']);key=(sample['backend'],sample['config'])
                    if not math.isfinite(value) or value<=0 or trial in groups[key]:raise ValueError('invalid sample')
                    groups[key].add(trial)
            observed={k:len(v) for k,v in groups.items()}
            if observed!=expected:raise ValueError(f'wrong candidate/trial counts: {observed}')
            if any(v!=set(range(meta['trials'])) for v in groups.values()):raise ValueError('trial ids are not contiguous')
        except Exception as exc:problems.append({'dataset':name,'problem':str(exc)})
    targeted={}
    for name in ('crossover-confirmation','medium-tail-confirmation','large-wins-confirmation'):
        p=root/name/'manifest.json'
        if not p.exists():targeted[name]={'complete':False};continue
        data=json.loads(p.read_text());runs=data.get('results',[]);bad=[];expected_targeted=set()
        for row in csv.reader((root/name/'configs.csv').open()):
            if row and not row[0].startswith('#'):expected_targeted.add((row[0],'/'.join(row[1:])))
        expected_targeted.add(('cusparse','CSR_ALG1'))
        for case in runs:
            q=root/name/f"{case['dataset']}-{case['seed']}.csv"
            if not q.exists():bad.append({'dataset':case['dataset'],'seed':case['seed'],'reason':'missing csv'});continue
            try:
                if case.get('file_sha256') and hashlib.sha256(q.read_bytes()).hexdigest()!=case['file_sha256']:raise ValueError('result hash mismatch')
                groups=collections.defaultdict(set)
                with q.open() as f:
                    for r in csv.DictReader(f):
                        v=float(r['us']);trial=int(r['trial']);key=(r['backend'],r['config'])
                        if not math.isfinite(v) or v<=0 or trial in groups[key]:raise ValueError('invalid sample')
                        groups[key].add(trial)
                if any(x!=set(range(21)) for x in groups.values()):raise ValueError('incomplete trials')
                if set(groups)!=expected_targeted:raise ValueError('candidate set mismatch')
            except Exception as exc:bad.append({'dataset':case['dataset'],'seed':case['seed'],'reason':str(exc)})
        targeted[name]={'complete':data.get('complete',False),'cases':len(data.get('cases',[])),'runs':len(runs),'failures':data.get('failures',[]),'problems':bad}
    out={'manifest_total':len(manifest),'status_counts':dict(counts),'sweep_problems':problems,'targeted':targeted}
    (root/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
    if problems or any(v.get('problems') for v in targeted.values()):raise SystemExit(1)
if __name__=='__main__':main()
