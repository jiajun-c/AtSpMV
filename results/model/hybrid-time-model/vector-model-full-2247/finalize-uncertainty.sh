#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/hybrid-time-model/vector-model-full-2247
pid=$(cat "$out/queue.pid")
while [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | rg -q 'vector-model-full-2247/run-after-80.sh'; do sleep 30; done
if [ ! -f "$out/analysis.json" ] || [ ! -f "$out/audit.json" ]; then echo 'full suite did not produce audited analysis'; exit 1; fi
python3 "$out/provenance/plan_speedup_uncertainty.py" "$out"
