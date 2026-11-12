#!/usr/bin/env python3
"""Check mixed policy routing, tuning gate and logged Plan choice."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

script=Path(__file__).resolve().parents[1]/'scripts/analysis/audit_model_fallback_policy.py'
with tempfile.TemporaryDirectory() as d:
    p=Path(d);(p/'raw').mkdir();(p/'status').mkdir()
    (p/'metadata.json').write_text(json.dumps({'selection':'auto_model_fallback_tuned','counterbalance':True,'retry_shared_storage_on_oom':True}))
    (p/'manifest.json').write_text(json.dumps([{'dataset':'small','nnz':10,'rows':2,'columns':2},{'dataset':'large','nnz':1000001,'rows':2,'columns':2}]))
    for name in ('small','large'):
        (p/'status'/f'{name}.json').write_text(json.dumps({'status':'ok'}))
        rows=[];log=[]
        for order in ('baseline_first','selection_first'):
            log.append(f'BENCH_ORDER,{order}')
            log.append(f'PLAN_OOM_RETRY,auto_model_fallback_tuned,{0 if name=="small" else 1}')
            if name=='small':
                log.append('MODEL_FALLBACK_POLICY,model');config='csr_vector/-1/-1/vw=4'
            else:
                log.extend(['MODEL_FALLBACK_POLICY,tuned','TUNING_BEGIN selected=csr_vector vector_width=4',
                            'TUNING,csr_vector,-1,-1,8,128,1,1024,4,10',
                            'TUNING,hybrid_dsm_atbsr,3,0,8,256,1,1024,0,9.6','TUNING_END'])
                config='csr_vector/-1/-1/vw=4'
            rows.append({'order':order,'backend':'auto_model_fallback_tuned','config':config})
        (p/'raw'/f'{name}.log').write_text('\n'.join(log)+'\n')
        with (p/'raw'/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    cmd=[sys.executable,str(script),str(p),'--assume-h100-default-scope']
    subprocess.run(cmd,check=True,capture_output=True)
    report=json.loads((p/'policy-audit.json').read_text());assert report['orders_checked']==4 and report['oom_retry_orders']=={'0':2,'1':2} and not report['issues']
    f=p/'raw/small.log';original=f.read_text();f.write_text(original.replace('auto_model_fallback_tuned,0','auto_model_fallback_tuned,1'))
    assert subprocess.run(cmd,capture_output=True).returncode!=0
    f.write_text(original)
    f=p/'raw/large.log';f.write_text(f.read_text().replace('0,9.6','0,9.4'))
    assert subprocess.run(cmd,capture_output=True).returncode!=0
print('model fallback policy audit passed')
