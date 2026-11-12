#!/usr/bin/env python3
"""A constant paired speedup must yield a degenerate bootstrap interval."""
import json
from pathlib import Path
import subprocess
import tempfile

root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as temporary:
    d=Path(temporary)
    data={'sample':[{'dataset':name,'predictions':{'deployed':{'speedup':1.0},'extended_deep':{'speedup':1.1}}}
                    for name in ('matrix_1','matrix_2','other_1')]}
    (d/'input.json').write_text(json.dumps(data))
    subprocess.run(['python3',str(root/'scripts/analysis/analyze_vector_replay_uncertainty.py'),
                    '--decisions',str(d/'input.json'),'--output',str(d/'result.json'),'--samples','100'],check=True,capture_output=True)
    result=json.loads((d/'result.json').read_text())['datasets']['sample']
    assert result['matrices']==3 and result['approximate_families']==2
    assert all(abs(value-1.1)<1e-12 for value in result['bootstrap_95pct_interval'])
print('vector replay uncertainty passed')
