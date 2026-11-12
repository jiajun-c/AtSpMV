#!/usr/bin/env python3
"""Evaluate frozen coefficients; never fits on the requested evaluation split."""
import argparse
import json
from pathlib import Path
import numpy as np
from fit_time_model import PARAMETERS, load_data, metrics, predict


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--data',required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--split',choices=['train','dev','test'],default='test')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    profile=json.loads(args.profile.read_text())
    data=load_data(args.manifest,args.data,[args.split])
    if args.split=='test':
        assert not set(data.matrix)&set(profile['train_matrices']+profile['dev_matrices'])
    coefficients=[profile['parameters'][key] for key in PARAMETERS]
    prediction=predict(data,coefficients)
    result=metrics(data,prediction)
    data=data.assign(prediction_us=prediction,relative_error=np.abs(prediction/data.measured_us-1))
    result['by_shape']={str(k):metrics(g,g.prediction_us.to_numpy()) for k,g in data.groupby('m')}
    result['by_mode']={str(k):metrics(g,g.prediction_us.to_numpy()) for k,g in data.groupby('balanced')}
    result['by_size']={str(k):metrics(g,g.prediction_us.to_numpy()) for k,g in data.groupby(
        np.floor(np.log10(data.nnz)).astype(int))}
    result['worst_candidates']=data.nlargest(12,'relative_error')[
        ['matrix','m','k','balanced','measured_us','prediction_us','relative_error']].to_dict('records')
    expected=[m for m in json.loads(Path(args.manifest).read_text())['matrices'] if m['split']==args.split]
    result['missing_or_failed']=[]
    for item in expected:
        if item['matrix'] not in set(data.matrix):
            meta=Path(args.data)/(item['matrix']+'.json')
            result['missing_or_failed'].append(json.loads(meta.read_text()) if meta.exists() else item)
    result['gate']={'mape_limit':.3,'p90_ape_limit':.5,
                    'passed':result['mape']<=.3 and result['p90_ape']<=.5 and not result['missing_or_failed']}
    if 'log_error_interval' in profile:
        lo,hi=profile['log_error_interval']
        low=prediction*np.exp(lo);high=prediction*np.exp(hi)
        result['interval_coverage']=float(np.mean((data.measured_us>=low)&(data.measured_us<=high)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    data.to_csv(args.output.with_suffix('.csv'),index=False)
    print(json.dumps({k:v for k,v in result.items() if not k.startswith('by_') and k!='worst_candidates'},indent=2))

if __name__=='__main__':main()
