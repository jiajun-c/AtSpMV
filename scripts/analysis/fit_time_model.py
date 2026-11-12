#!/usr/bin/env python3
"""Calibrate a cache/occupancy/tail model with matrix-disjoint validation."""
import argparse
import json
import hashlib
import scipy
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

PARAMETERS = ['launch_us', 'dram_eff', 'l2_factor', 'tile_ns_1', 'tile_ns_2',
              'tile_ns_4', 'tile_ns_8', 'tile_ns_16', 'tile_ns_32',
              'tile_cycles', 'warp_cycles', 'shuffle_cycles', 'atomic_avg_ns',
              'atomic_serial_ns', 'saturation_warps', 'cache_capacity_fraction', 'cache_transition']
INITIAL = [1.5, .7, 8, 80,80,80,80,80,80, 24,32,8, 2,12,16,.85,8]
LOWER = [.05,.05,1, 1,1,1,1,1,1, 1,0,0, 0,0, 2,.1,1]
UPPER = [8,1,100, 300,300,300,300,300,300, 256,256,64, 100,100, 64,2,30]


def load_data(manifest, directory, splits=('train','dev')):
    tables=[]
    for item in json.loads(Path(manifest).read_text())['matrices']:
        if item['split'] not in splits:
            continue
        meta=Path(directory)/(item['matrix']+'.json')
        if not meta.exists() or json.loads(meta.read_text())['status']!='ok':
            continue
        table=pd.read_csv(Path(directory)/(item['matrix']+'.csv'))
        for field, baseline in [('cap', 8), ('warps_per_block', 4), ('prefetch', 0), ('cta_reduce', 0)]:
            if field in table and not table[field].eq(baseline).all():
                raise ValueError(f"{item['matrix']}: v2 calibration only supports baseline kernel configuration ({field})")
        table['split']=item['split'];table['family']=item['family']
        if 'cache_mode' not in table: table['cache_mode']='warm'
        tables.append(table)
    return pd.concat(tables,ignore_index=True)


def predict(frame, parameters):
    p=np.asarray(parameters)
    m=frame.m.to_numpy();k=frame.k.to_numpy();bal=frame.balanced.to_numpy()
    tiles=frame.tiles.to_numpy();warps=frame.warps.to_numpy();rows=frame.rows.to_numpy()
    sm=frame.sm_count.to_numpy()
    active=np.minimum(sm,np.ceil(warps/4))
    per_sm=tiles/active
    concurrent=np.minimum(frame.resident_warps,warps/active)
    saturation=np.minimum(1,concurrent/p[14])
    # DRAM versus L2 residency is a soft transition, rather than a hard cutoff.
    footprint=tiles*32*(8+frame.index_bytes)+8*(frame.cols+rows)
    ratio=footprint/np.maximum(1,frame.l2_bytes)
    miss=1/(1+np.exp(np.clip(-p[16]*(ratio-p[15]),-50,50)))
    if 'cache_mode' in frame:
        miss=np.where(frame.cache_mode=='cold',1.0,miss)
    peak_bw=2*frame.memory_clock_khz*(frame.memory_bus_bits/8)/1e6
    traffic=tiles*32*(8+frame.index_bytes)+8*frame.nnz+warps*(8+4*bal)+rows*12
    memory_ns=traffic/peak_bw*(miss/p[1]+(1-miss)/p[2])*sm/active/saturation
    fp32=np.where((frame.cc_major==8)&(frame.cc_minor==0),64,128)
    fp64=fp32/frame.fp64_ratio
    cycles=np.maximum(p[9],32/fp64)
    issue_ns=(per_sm*cycles+warps/active*(p[10]+np.log2(k)*p[11]))/(frame.clock_khz/1e6)
    longest=np.where(bal,np.minimum(frame.max_tiles,8),frame.max_tiles)
    tile_cost=p[3+np.log2(m).astype(int)]
    dependency_ns=np.maximum(per_sm/concurrent,longest)*tile_cost
    atomic_ns=bal*(p[12]*warps*m/active + p[13]*np.ceil(frame.max_tiles/8)*m/32)
    return np.asarray(p[0]+(np.maximum.reduce([memory_ns,issue_ns,dependency_ns])+atomic_ns)/1000)


def metrics(frame,prediction):
    actual=frame.measured_us.to_numpy()
    ape=np.abs(prediction/actual-1)
    regrets=[]
    out=frame.assign(prediction_us=prediction)
    for _,group in out.groupby('matrix'):
        chosen=group.loc[group.prediction_us.idxmin(),'measured_us']
        regrets.append(chosen/group.measured_us.min()-1)
    return dict(matrices=frame.matrix.nunique(),candidates=len(frame),
                mape=float(np.mean(ape)),median_ape=float(np.median(ape)),
                p90_ape=float(np.quantile(ape,.9)),max_ape=float(np.max(ape)),
                median_regret=float(np.median(regrets)),p90_regret=float(np.quantile(regrets,.9)),
                timing_p90_relative_spread=float(np.quantile((frame.p90_us-frame.p10_us)/actual,.9)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--data',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=load_data(args.manifest,args.data)
    train=data[data.split=='train']; dev=data[data.split=='dev']
    def residual(p):
        return np.log(predict(train,p)/train.measured_us.to_numpy())
    fit=least_squares(residual,INITIAL,bounds=(LOWER,UPPER),max_nfev=500,
                      loss='soft_l1',f_scale=.3,x_scale='jac')
    hardware_columns=['sm_count','l2_bytes','cc_major','cc_minor','fp64_ratio']
    hardware_rows=data[hardware_columns].drop_duplicates()
    if len(hardware_rows)!=1:
        raise ValueError('Fit one hardware profile at a time')
    hardware={k:int(v) for k,v in hardware_rows.iloc[0].items()}
    hardware['fp64_lanes_per_cycle']=(64 if hardware['cc_major']==8 and hardware['cc_minor']==0 else 128)/hardware['fp64_ratio']
    if data.cache_mode.nunique()!=1: raise ValueError('Fit one cache regime at a time')
    cache_mode=data.cache_mode.iloc[0]
    report=dict(schema=2,hardware=hardware,cache_mode=cache_mode,
                manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                numpy_version=np.__version__,scipy_version=scipy.__version__,
                timing_protocol=cache_mode+'_cuda_graph_median_9_batches',
                parameters=dict(zip(PARAMETERS,fit.x.tolist())),
                train=metrics(train,predict(train,fit.x)),dev=metrics(dev,predict(dev,fit.x)),
                train_matrices=sorted(train.matrix.unique()),dev_matrices=sorted(dev.matrix.unique()),
                log_error_interval=np.quantile(np.log(dev.measured_us.to_numpy()/predict(dev,fit.x)),[.05,.95]).tolist())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    data.assign(prediction_us=predict(data,fit.x)).to_csv(args.output.with_suffix('.csv'),index=False)
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('matrices')},indent=2))

if __name__=='__main__':
    main()
