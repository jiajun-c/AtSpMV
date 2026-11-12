#!/usr/bin/env python3
"""Compare cluster sizes, peer ranks, and atomic controls with reproducible provenance."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exe',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--iterations',type=int,nargs='+',default=[1024,2048])
    ap.add_argument('--seeds',type=int,nargs='+',default=[202610271,202610272,202610273])
    ap.add_argument('--trials',type=int,default=15)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[2]
    gpu=['nvidia-smi','--query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used','--format=csv']
    meta={'source_sha256':sha(repo/'bench/dsm_cluster_sweep.cu'),'script_sha256':sha(repo/'scripts/analysis/benchmark_dsm_cluster_sweep.py'),
          'binary':str(args.exe.resolve()),'binary_sha256':sha(args.exe),
          'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
          'iterations':args.iterations,'seeds':args.seeds,'trials':args.trials,
          'method':'32 calls per CUDA graph; 3 warmup rounds and randomly interleaved candidates each round; p10/median/p90 from trials. Clock64 samples refer to the last graph node; cudaEvent time is per graph node.',
          'timing_limits':'Dependent loops report cycles/iteration including dependent-address or value arithmetic. Single atomic includes clock/branch/issue and need not await atomic completion. Returnless RED clock is not completion latency. DSM round times mbarrier transaction and rank0 combine after initial cluster.sync; event time includes initialization. Atomic_plain kernels have no cluster barrier. Independent output addresses are 128 bytes apart. All hot caches; no clocks locked.',
          'validation':'Every candidate checked via direct launch and graph replay. Each kernel opts into nonportable size 16 and is tested for active-cluster occupancy before use. Number of distinct physical SM IDs from final graph node is recorded.',
          'cluster_size':'2/4/8/16; 16 is nonportable and omitted explicitly if any tested kernel cannot run it. Remote peer is rank 1 or last rank. Single cluster per kernel; occupancy numbers are CUDA API maxima, not achieved occupancy.',
          'fixed16':'dsm_async_fixed16 keeps 16 warp partials per cluster, with 256/128/64/32 threads per CTA at cluster 2/4/8/16. The broadcast still sends to all C CTAs.'}
    runs=[]
    for n in args.iterations:
        for seed in args.seeds:
            before=subprocess.check_output(gpu,text=True)
            path=args.output/f'iter{n}-seed{seed}.csv'
            with path.open('w') as f:
                subprocess.run([str(args.exe.resolve()),str(n),str(args.trials),str(seed)],stdout=f,check=True)
            with path.open() as f:rows=list(csv.DictReader(f))
            expected=128 if any(r['cluster']=='16' and r['kind']=='dsm_async_round' for r in rows) else 100
            if len(rows)!=expected:raise RuntimeError(f'expected {expected} candidates, got {len(rows)}')
            after=subprocess.check_output(gpu,text=True)
            runs.append({'iterations':n,'seed':seed,'csv':path.name,'gpu_before':before,'gpu_after':after,'candidates':len(rows),'cluster16_supported':expected==128})
            (args.output/'runs.json').write_text(json.dumps(runs,indent=2)+'\n')
            print('completed',n,seed,flush=True)
    meta['runs']=len(runs)
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    result=[]
    for n in args.iterations:
        groups={}
        for seed in args.seeds:
            with (args.output/f'iter{n}-seed{seed}.csv').open() as f:
                for r in csv.DictReader(f):groups.setdefault((r['kind'],r['cluster'],r['peer_rank'],r['threads'],r['iterations']),[]).append(r)
        for (kind,cluster,peer,threads,count),rows in groups.items():
            if len(rows)!=len(args.seeds):raise RuntimeError('inconsistent candidate set')
            median=lambda k:statistics.median(float(r[k]) for r in rows)
            record={'kind':kind,'cluster':int(cluster),'peer_rank':int(peer),'threads':int(threads),'active_clusters':int(rows[0]['active_clusters']),
                    'loop_length':int(count),'sweep_iterations':n,
                    'median_cycles_per_iter':median('median_cycles_per_iter'),
                    'min_run_p10_cycles':min(float(r['p10_cycles']) for r in rows),
                    'max_run_p90_cycles':max(float(r['p90_cycles']) for r in rows),
                    'median_max_cta_cycles_per_iter':median('median_max_cta_cycles_per_iter'),
                    'median_kernel_us':median('median_kernel_us'),
                    'min_sms':min(int(r['min_distinct_sms']) for r in rows),
                    'max_sms':max(int(r['max_distinct_sms']) for r in rows)}
            if kind.startswith('atomic_') and int(count)>1:
                record['effective_giga_ops_per_second']=int(cluster)*int(count)/(record['median_kernel_us']*1000)
            else:record['effective_giga_ops_per_second']=None
            result.append(record)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (args.output/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)


if __name__=='__main__':main()
