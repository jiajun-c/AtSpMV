#!/usr/bin/env python3
"""Check frozen per-matrix configuration and two-order result aggregation."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as temporary:
    d=Path(temporary); (d/'matrices/m0').mkdir(parents=True)
    (d/'matrices/m0/m0.mtx').write_text('fixture')
    replay=d/'replay.csv'
    with replay.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','split','selected_medium','medium_tile','medium_parts','existing_config']);w.writeheader()
        for i in range(38): w.writerow({'dataset':f'm{i}','split':'holdout','selected_medium':'True','medium_tile':'1','medium_parts':'4','existing_config':'null'})
    hashes={f'm{i}':hashlib.sha256(b'fixture').hexdigest() for i in range(38)}
    (d/'hashes.json').write_text(json.dumps(hashes))
    mock=d/'mock.py'
    mock.write_text('#!/usr/bin/env python3\nimport csv,sys\n'
                    'print("backend,config,trial,us")\n'
                    'rows=list(csv.reader(open(sys.argv[2])))+[["cusparse"]]\n'
                    'for r in rows:\n'
                    ' if r[0] != "cusparse": print(f"PLAN_PREP,{r[0]},100",file=sys.stderr)\n'
                    'for r in rows:\n'
                    ' for i in range(int(sys.argv[4])):\n'
                    '  time = 1 if r[0] == "hybrid_dsm_atbsr" else 2\n'
                    '  config = "atbsr/3/0/8/128/1/1024/vw=0" if r[0] == "auto_vector_model_v2" else "fixture"\n'
                    '  print(f"{r[0]},{config},{i},{time}")\n')
    mock.chmod(0o755)
    runner=root/'scripts/analysis/benchmark_medium_unified.py'
    command=['python3',str(runner),'--replay',str(replay),'--exe',str(mock),'--matrix-root',str(d/'matrices'),
             '--output',str(d/'out'),'--idle-helper',str(mock),'--expected-hashes',str(d/'hashes.json'),'--gpu-index','0','--trials','2','--case','m0']
    subprocess.run(command,check=True,capture_output=True)
    subprocess.run(['python3',str(root/'scripts/analysis/analyze_medium_unified.py'),str(d/'out'),'--replay',str(replay)],check=True,capture_output=True)
    report=json.loads((d/'out/analysis.json').read_text())
    assert report['cases']==1 and report['status_counts']['pending']==37 and report['speedup_geomean']==2
    assert report['amortization']['10000']['geomean_speedup'] > 1
    assert 'hybrid_dsm_atbsr,1,0,4,256,4,128' in (d/'out/configs/m0-baseline_first.csv').read_text()
    (d/'matrices/m0/m0.mtx').write_text('changed')
    assert subprocess.run(command,capture_output=True).returncode != 0
print('medium unified benchmark passed')
