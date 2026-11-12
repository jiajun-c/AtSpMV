#!/usr/bin/env python3
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

script=Path(__file__).resolve().parents[1]/'scripts/analysis/analyze_large_scope_risk.py'
with tempfile.TemporaryDirectory() as d:
    p=Path(d);(p/'index.csv').write_text('dataset\na\nb\n');(p/'history.csv').write_text('dataset,best_nonhybrid_us,atbsr_default_us\na,1,2\nb,2,2\n')
    cmd=[sys.executable,str(script),'--large-index',str(p/'index.csv'),'--historical-analysis',str(p/'history.csv'),'--in-scope-count','2','--output',str(p/'out.json')]
    subprocess.run(cmd,check=True,capture_output=True);r=json.loads((p/'out.json').read_text())
    assert math.isclose(r['historical_large_default_vs_fixed_best']['geomean'],math.sqrt(.5))
    assert math.isclose(r['required_in_scope_geomean_for_combined_one_if_historical_large_ratio_held'],math.sqrt(2))
    (p/'index.csv').write_text('dataset\na\nc\n');assert subprocess.run(cmd,capture_output=True).returncode!=0
print('large-scope risk analysis passed')
