"""Rebuild the final native-AtBSR comparison from immutable trial outputs."""
import csv
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
report={'applications': {}, 'kernels': [], 'profile': []}
for case in ('confirmation-r7', 'confirmation-unbound-r7'):
    manifest=json.loads((ROOT/case/'manifest.json').read_text())
    summary=json.loads((ROOT/case/'summary.json').read_text())
    assert not manifest['persist_x'] and not summary['persist_x']
    assert summary['validated_processes']==24
    report['applications'][case]=summary
for refine in (5,7):
    rows=[]
    for rnd in range(3):
        data=list(csv.DictReader((ROOT/f'cusparse-algorithms/r{refine}-{rnd}.csv').open()))
        assert len(data)==31 and all(float(r['max_backward_error'])<1e-12 for r in data)
        rows.extend(data)
    for name in ('cusparse', 'cusparse-default', 'cusparse-alg2', 'atbsr-w4-d2', 'compressed-w4', 'grouped-w4'):
        values=[float(r['median_us']) for r in rows if r['candidate']==name]
        report['kernels'].append({'refine': refine, 'candidate': name, 'median_us': statistics.median(values)})
for backend in ('atbsr','atbsr-compressed','atbsr-grouped'):
    meta=json.loads((ROOT/f'profile/{backend}.json').read_text());assert meta['returncode']==0 and '-persist-x' not in meta['command']
    lines=(ROOT/f'profile/{backend}.csv').read_text().splitlines()
    start=next(i for i,line in enumerate(lines) if line.startswith('"ID"'))
    values={r['Metric Name']: float(r['Metric Value']) for r in csv.DictReader(lines[start:])}
    report['profile'].append({'backend':backend, **values})
out=ROOT/'report';out.mkdir(exist_ok=True)
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
with (out/'kernels.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(report['kernels'][0]));writer.writeheader();writer.writerows(report['kernels'])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,4))
names=('cusparse-default','atbsr-w4-d2','compressed-w4','grouped-w4')
labels=('cuSPARSE\nDEFAULT','Original\nAtBSR','Per-tile\ncompressed','Grouped\nAtBSR')
values=[next(r['median_us'] for r in report['kernels'] if r['refine']==7 and r['candidate']==n) for n in names]
axes[0].bar(range(4),values,color=['#e79b38','#8a8a8a','#9bbcd1','#b44343'])
axes[0].set_xticks(range(4),labels);axes[0].set_ylabel('SpMV kernel time (us)')
axes[0].set_title('Default L2 policy; no persisting window')
data=report['applications']['confirmation-unbound-r7']['backends']
backends=('cusparse','atbsr','atbsr-grouped')
values=[data[b]['median_total_s'] for b in backends]
axes[1].bar(range(3),values,color=['#e79b38','#8a8a8a','#b44343'])
axes[1].set_xticks(range(3),['cuSPARSE\nALG1','Original\nAtBSR','Grouped\nAtBSR'])
axes[1].set_ylabel('Application compute time (s)')
axes[1].set_title('Default CPU scheduling; eight paired rounds')
for i,v in enumerate(values):axes[1].text(i,v*.5,f'{v:.3f}',ha='center',color='white')
fig.tight_layout();fig.savefig(out/'native-comparison.png',dpi=200);fig.savefig(out/'native-comparison.pdf');plt.close(fig)
print(json.dumps(report['applications']['confirmation-unbound-r7']['backends']['atbsr-grouped']['vs_cusparse'],indent=2))
