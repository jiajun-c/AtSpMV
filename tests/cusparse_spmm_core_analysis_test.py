#!/usr/bin/env python3
"""Check paired-order calculations and missing-case handling."""
import csv
import json
from pathlib import Path
import subprocess
import tempfile

root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as tmp:
    out=Path(tmp);(out/'cases').mkdir()
    for order,csr,ell in (('forward',2,4),('reverse',8,8)):
        common={'m':'1024','k':'1024','n':'8','nnz':'100','block':'16',
                'blocks_per_row':'2','requested_density':'5','status':'ok'}
        rows=[dict(common,method='csr_alg2',median_us=str(csr)),
              dict(common,method='blocked_ell',median_us=str(ell))]
        (out/'cases'/f'b2-d5-n8-{order}.json').write_text(json.dumps({'returncode':0,'rows':rows}))
    subprocess.run(['python3',str(root/'scripts/analysis/analyze_cusparse_spmm_cores.py'),str(out)],
                   check=True,capture_output=True)
    with (out/'comparison.csv').open(newline='') as f:
        data=list(csv.DictReader(f))
    entry=next(r for r in data if r['blocks_per_row']=='2' and r['density_percent']=='5' and r['n']=='8' and r['method']=='csr_alg2')
    assert entry['status']=='ok' and abs(float(entry['ell_over_csr'])-2**.5)<1e-12
    summary=json.loads((out/'summary.json').read_text())
    assert summary['comparisons']==36 and summary['valid']==1 and summary['csr_more_than_5pct_faster']==1
print('cuSPARSE SpMM paired analysis passed')
