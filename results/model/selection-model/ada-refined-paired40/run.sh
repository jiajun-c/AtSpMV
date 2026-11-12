#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/selection-model/ada-refined-paired40
while true; do
 python3 "$out/provenance/benchmark_plan_selection.py" --exe "$out/provenance/spmv_hybrid_validation" --index "$out/index.csv" --matrix-root /staff/chengjiajun/workspace/database/suitsparse --output "$out" --gpu-index 1 --selection auto_model_refined --compare-full-tune --counterbalance --internal-graph-events >> "$out/runner.log" 2>&1
 rc=$?
 [ "$rc" -eq 75 ] || break
done
if [ "$rc" -eq 0 ]; then python3 "$out/provenance/analyze_plan_selection.py" "$out" > "$out/summary.log" 2>&1; fi
printf 'runner exit code: %s\n' "$rc" >> "$out/runner.log"
exit "$rc"
