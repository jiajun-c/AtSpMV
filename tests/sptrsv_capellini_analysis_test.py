#!/usr/bin/env python3
import json,subprocess,sys,tempfile
from pathlib import Path
base=Path(__file__).resolve().parents[1]/'scripts/analysis'
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);(root/'cases').mkdir();(root/'manifest.json').write_text(json.dumps({'names':['factor']}))
    for rnd in range(2):
        plan='mode,backend,rows,solve_us,max_error\nmodel-host,cached,128,5,1e-13\n'
        cap='time_cuda_solve = 0.01 ms\nVALIDATION finite=1 max_error=1e-13 reference_scale=1.49\n'
        (root/'cases'/f'factor-plan-{rnd}.json').write_text(json.dumps({'returncode':0,'stdout':plan}))
        (root/'cases'/f'factor-tc-{rnd}.json').write_text(json.dumps({'returncode':0,'stdout':cap}))
    subprocess.run([sys.executable,str(base/'analyze_sptrsv_capellini.py'),str(root),'--parser',str(base/'analyze_sptrsv_tc.py')],check=True,capture_output=True,text=True)
    result=json.loads((root/'summary.json').read_text())
    assert result['geomean_capellini_over_plan']==2 and 'geomean_tc_over_plan' not in result
    assert 'adapted Capellini' in result['caveat']
    assert result['capellini_warmup_solves']==0
    subprocess.run([sys.executable,str(base/'analyze_sptrsv_capellini.py'),str(root),'--parser',str(base/'analyze_sptrsv_tc.py'),'--warmup-count','5'],check=True,capture_output=True,text=True)
    assert json.loads((root/'summary.json').read_text())['capellini_warmup_solves']==5
    subprocess.run([sys.executable,str(base/'analyze_sptrsv_capellini.py'),str(root),'--parser',str(base/'analyze_sptrsv_tc.py'),'--warmup-count','5','--poison-outside-timing'],check=True,capture_output=True,text=True)
    assert json.loads((root/'summary.json').read_text())['output_poisoned_outside_timing'] is True
