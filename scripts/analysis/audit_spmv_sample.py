#!/usr/bin/env python3
"""Audit the frozen sample, process outcomes and complete raw candidate coverage."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from benchmark_dsm_atbsr_full import sha, write_json
from analyze_spmv_sample import PLANS, CUSPARSE


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);a=parser.parse_args();p=a.output
    manifest=json.loads((p/'manifest.json').read_text());metadata=json.loads((p/'metadata.json').read_text())
    assert len(manifest['matrices'])==100
    assert sha(p/'manifest.json')==metadata['manifest_sha256']
    assert sha(p/'provenance/spmv_hybrid_validation')==metadata['exe_sha256']
    assert sha(p/'provenance/scripts/analysis/benchmark_optimization_ablation.py')==metadata['runner_sha256']
    migration=json.loads((p/'runner-migration.json').read_text())
    for digest,source in migration['runner_sources'].items():assert sha(p/source)==digest
    assert len({r['dataset'] for r in manifest['matrices']})==100
    strata=Counter((r['size_bin'],r['has_long_row']) for r in manifest['matrices'])
    assert all(strata[b,False]==13 and strata[b,True]==12 for b in range(4))
    assert max(Counter(r['approximate_family'] for r in manifest['matrices']).values())<=3
    pending=[];failures=[];count=0;ok=0
    for item in manifest['matrices']:
        name=item['dataset'];status_file=p/'status'/f'{name}.json'
        if not status_file.exists():pending.append(name);continue
        status=json.loads(status_file.read_text())
        if status['status']=='running':pending.append(name);continue
        if status['status']!='ok':failures.append({'dataset':name,'status':status['status']});continue
        assert status['runner_sha256'] in migration['runner_sources']
        assert len(status['matrix_sha256'])==64 and len(status['orders'])==2
        for order in (0,1):
            record=status['orders'][order];assert record['returncode']==0 and record['reason'] is None
            raw=list(csv.DictReader((p/'raw'/f'{name}-{order}.csv').open()))
            counts=Counter(r['backend'] for r in raw)
            assert set(counts)==set(PLANS+CUSPARSE)
            assert all(n==manifest['trials'] for n in counts.values())
            assert len({(r['backend'],r['trial']) for r in raw})==len(raw)
            count+=len(raw)
        ok+=1
    audit={'status':'passed' if ok==100 and not pending and not failures else 'incomplete',
           'successful':ok,'pending':pending,'failures':failures,'raw_timings':count,
           'full_vector_checks':count//manifest['trials']*3,
           'verified':'sample strata/family cap, frozen binary/runner/manifest hashes, input hash records, per-process return codes, backend and trial coverage. Numerical assertions execute inside the benchmark.'}
    write_json(p/'coverage-audit.json',audit);print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
