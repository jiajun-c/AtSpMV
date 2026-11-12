#!/usr/bin/env python3
"""Recheck frozen hybrid and competing configurations on a free Hopper GPU."""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    with path.open('rb') as f:
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
        return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exe', type=Path, required=True)
    ap.add_argument('--matrix-root', type=Path, required=True)
    ap.add_argument('--config-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--names', nargs='+', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[202610271,202610272,202610273])
    ap.add_argument('--trials', type=int, default=21)
    args = ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    gpu = ['nvidia-smi','--query-gpu=index,uuid,name,utilization.gpu,memory.used','--format=csv']
    files = ['bench/spmv_hybrid_validation.cu','src/spmvPlan.cu','src/spmvPlanDsm.cu',
             'include/spmvPlan.h','include/detail/spmvPlanDsm.h','include/dsmLongRow.cuh',
             'src/atspmvProduct.cu','scripts/analysis/benchmark_dsm_atbsr_hybrid.py']
    metadata = {'gpu_before':subprocess.check_output(gpu,text=True),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                'exe':str(args.exe.resolve()),'exe_sha256':digest(args.exe),
                'sources':{p:digest(ROOT/p) for p in files},'inputs':{},'configs':{},
                'seeds':args.seeds,'trials':args.trials,
                'method':'Fixed exploratory shortlist; direct correctness with two nonuniform x vectors and poisoned y, plus CUDA Graph correctness. Three warmup and randomly interleaved timed batches; 8 calls per graph, warm graph before every timed graph, cudaEvent time per call. Setup, preprocessing and graph creation excluded.',
                'selection':'Per dataset, take the median trial for each frozen configuration and seed, then median across seeds. Report best measured configuration in each backend and best measured non-hybrid. Exploratory, not an independent holdout.'}
    all_results=[]
    for name in args.names:
        matrix=args.matrix_root/name/(name+'.mtx');cfg=args.config_dir/(name+'-configs.csv')
        metadata['inputs'][name]={'path':str(matrix.resolve()),'sha256':digest(matrix)}
        metadata['configs'][name]={'path':str(cfg.resolve()),'sha256':digest(cfg)}
        grouped=collections.defaultdict(list)
        for seed in args.seeds:
            out=args.output/f'{name}-{seed}.csv';err=args.output/f'{name}-{seed}.log'
            with out.open('w') as stdout,err.open('w') as stderr:
                subprocess.run([str(args.exe.resolve()),str(matrix),str(cfg),str(seed),str(args.trials)],stdout=stdout,stderr=stderr,check=True)
            run=collections.defaultdict(list)
            with out.open() as f:
                for r in csv.DictReader(f):run[(r['backend'],r['config'])].append(float(r['us']))
            for key,v in run.items():
                if len(v)!=args.trials:raise RuntimeError(f'missing samples {name} {seed} {key}')
                grouped[key].append(statistics.median(v))
        by_kind={}
        for (kind,config),values in grouped.items():
            if len(values)!=len(args.seeds):raise RuntimeError('missing seed')
            obj={'config':config,'median_us':statistics.median(values),'per_seed_us':values}
            if kind not in by_kind or obj['median_us']<by_kind[kind]['median_us']:by_kind[kind]=obj
        hybrid=by_kind['hybrid_dsm_atbsr'];baseline=min((v['median_us'],k) for k,v in by_kind.items() if k!='hybrid_dsm_atbsr')
        all_results.append({'dataset':name,'hybrid':hybrid,'backends':by_kind,'best_nonhybrid':baseline[1],
                            'best_nonhybrid_us':baseline[0],'speedup_vs_best_nonhybrid':baseline[0]/hybrid['median_us']})
    metadata['gpu_after']=subprocess.check_output(gpu,text=True)
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(all_results,indent=2)+'\n')
    with (args.output/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','hybrid_us','hybrid_config','best_nonhybrid','best_nonhybrid_us','speedup','cusparse_us','atbsr_us'])
        for r in all_results:w.writerow([r['dataset'],r['hybrid']['median_us'],r['hybrid']['config'],r['best_nonhybrid'],r['best_nonhybrid_us'],r['speedup_vs_best_nonhybrid'],r['backends']['cusparse']['median_us'],r['backends']['atbsr']['median_us']])
if __name__=='__main__':main()
