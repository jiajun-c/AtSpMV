#!/usr/bin/env python3
"""Plot remote DSM access and communication across cluster sizes."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    ap.add_argument('--iterations',type=int,default=2048)
    args=ap.parse_args()
    rows=json.loads((args.directory/'summary.json').read_text())
    sizes=sorted({r['cluster'] for r in rows if r['kind']=='dsm_async_round'})
    def get(kind,c,peer=-1,length=1):
        return next(r for r in rows if r['kind']==kind and r['cluster']==c and r['peer_rank']==peer and r['loop_length']==length and r['sweep_iterations']==args.iterations)
    fig,axs=plt.subplots(1,2,figsize=(12,4.7))
    for peer_kind,style in [('rank 1','o-'),('last rank','s--')]:
        chosen=[get('remote_dsm_load',c,1 if peer_kind=='rank 1' else c-1,args.iterations) for c in sizes]
        vals=np.array([r['median_cycles_per_iter'] for r in chosen])
        err=np.array([[v-r['min_run_p10_cycles'] for v,r in zip(vals,chosen)], [r['max_run_p90_cycles']-v for v,r in zip(vals,chosen)]])
        axs[0].errorbar(sizes,vals,yerr=err,fmt=style,capsize=3,label=peer_kind)
    axs[0].set_title('Remote 32-bit dependent read');axs[0].set_ylabel('SM clock cycles / read')
    for kind,label,style in [('dsm_async_round','4 warps per CTA','o-'),('dsm_async_fixed16','16 partials per cluster','s--')]:
        chosen=[get(kind,c) for c in sizes];vals=np.array([r['median_cycles_per_iter'] for r in chosen])
        err=np.array([[v-r['min_run_p10_cycles'] for v,r in zip(vals,chosen)], [r['max_run_p90_cycles']-v for v,r in zip(vals,chosen)]])
        axs[1].errorbar(sizes,vals,yerr=err,fmt=style,capsize=3,label=label)
    axs[1].set_title('DSM async communication round');axs[1].set_ylabel('rank-0 SM clock cycles / round')
    for ax in axs:
        ax.set_xscale('log',base=2);ax.set_xticks(sizes,labels=[str(x) for x in sizes]);ax.set_xlabel('Cluster size (CTAs)')
        ax.grid(alpha=.25);ax.legend(fontsize=9)
    fig.suptitle('H100 PCIe: cluster-size experiment',fontsize=13)
    fig.text(.5,.018,'Warm-cache measurements; spans show the range of per-run p10–p90, not confidence intervals.\nThe fixed-16-partial control changes threads/CTA and still broadcasts to all CTAs.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.08,1,.94]);fig.savefig(args.directory/'cluster-scaling.png',dpi=180)
    fig.savefig(args.directory/'cluster-scaling.svg');plt.close(fig)


if __name__=='__main__':main()
