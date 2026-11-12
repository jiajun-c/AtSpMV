#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/hybrid-time-model/shared-row-sort-paired80
prev=results/model/hybrid-time-model/shared-row-sort-h100
pid=$(cat "$prev/correctness-queue.pid")
while [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | rg -q 'shared-row-sort-h100/run-correctness-after.sh'; do sleep 30; done
if [ ! -f "$prev/direct.log" ] || [ ! -f "$prev/memcheck.log" ]; then echo 'H100 correctness did not complete'; exit 1; fi
while true; do
    python3 "$out/provenance/benchmark_plan_selection.py" --exe "$out/provenance/spmv_hybrid_validation" --index "$out/index.csv" --matrix-root /staff/chengjiajun/workspace/database/suitsparse --output "$out" --gpu-index 3 --monitor-resident --selection auto_tuned --counterbalance --extended-baselines >> "$out/runner.log" 2>&1
    rc=$?
    if [ "$rc" -ne 75 ]; then break; fi
done
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
python3 "$out/provenance/audit_plan_sweep.py" "$out" || exit $?
python3 "$out/provenance/audit_tuned_choice.py" "$out" || exit $?
python3 "$out/provenance/analyze_plan_selection.py" "$out" || exit $?
python3 "$out/provenance/analyze_plan_amortization.py" "$out" || exit $?
