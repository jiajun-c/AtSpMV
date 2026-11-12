#!/usr/bin/env python3
"""Plot independent H100 dependent-read and FP64 atomic cycle measurements."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--iterations',type=int,default=4096)
    args=parser.parse_args()
    data=json.loads((args.directory/'summary.json').read_text())
    def row(kind,cluster):
        return next(x for x in data if x['sweep_iterations']==args.iterations and x['kind']==kind and x['cluster']==cluster and x['loop_length']==args.iterations)
    panels=[('Dependent 32-bit read chain', [('Local shared','local_shared_load',2),('Remote DSM','remote_dsm_load',2),('Global .cg (L2)','global_l2_load_cg',2)]),
            ('Returned FP64 atomic dependency chain', [('Global FP64','atomic_plain_return_independent',1),('Local shared FP64','local_shared_fp64_atomic',2),('Remote DSM FP64','remote_dsm_fp64_atomic',2)])]
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for ax,(title,cases) in zip(axes,panels):
        rows=[row(k,c) for _,k,c in cases]; vals=[x['median_cycles_per_iter'] for x in rows]
        err=[[v-x['min_run_p10_cycles'] for v,x in zip(vals,rows)],[x['max_run_p90_cycles']-v for v,x in zip(vals,rows)]]
        ax.barh(range(len(rows)),vals,color=['#32769d','#c78335','#6579a2'],xerr=err,capsize=3)
        ax.set_yticks(range(len(rows)),[label for label,_,_ in cases]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('SM clock cycles / operation');ax.set_title(title)
        ax.set_xlim(0,max(vals)*1.20)
        for i,v in enumerate(vals):ax.text(v+max(vals)*.025,i,f'{v:.1f}',va='center',fontsize=10)
    fig.suptitle('H100 PCIe: single active thread, warm working set',fontsize=13)
    fig.text(.5,.015,'Read and atomic panels have different operations; cycles include loop dependencies. Shared FP64 uses the generated generic/CAS path.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.055,1,.94]);fig.savefig(args.directory/'dependent-cycles.png',dpi=180)
    fig.savefig(args.directory/'dependent-cycles.svg');plt.close(fig)


if __name__=='__main__':main()
