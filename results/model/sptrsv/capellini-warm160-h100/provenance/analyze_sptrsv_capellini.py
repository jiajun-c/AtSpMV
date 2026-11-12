#!/usr/bin/env python3
"""Apply shared paired validation and label results as adapted Capellini."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--parser',type=Path,required=True);p.add_argument('--warmup-count',type=int,default=0);a=p.parse_args()
if a.warmup_count<0:p.error('--warmup-count must be nonnegative')
subprocess.run([sys.executable,str(a.parser),str(a.directory),'--aligned'],check=True,capture_output=True,text=True)
path=a.directory/'summary.json';summary=json.loads(path.read_text());summary['geomean_capellini_over_plan']=summary.pop('geomean_tc_over_plan')
summary['caveat']='locally adapted Capellini mix and Plan CUDA 12.6; identical padded unit-diagonal factors; synchronized host timing includes per-solve resets and excludes preprocessing; two rounds, invalid omitted; tc-named raw columns refer to Capellini'
summary['capellini_warmup_solves']=a.warmup_count
path.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
