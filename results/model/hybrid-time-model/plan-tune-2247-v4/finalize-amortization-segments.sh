#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/hybrid-time-model/plan-tune-2247-v4
pid=1048286
while [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | rg -q 'finalize_plan_sweep.py'; do sleep 30; done
if [ ! -f "$out/completion.json" ]; then echo 'Sweep finalizer did not complete'; exit 1; fi
python3 "$out/provenance/analyze_amortization_segments.py" "$out" --output "$out/amortization-segments.json"
