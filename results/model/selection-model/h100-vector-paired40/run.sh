#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/selection-model/h100-vector-paired40
while true; do
 python3 "$out/provenance/benchmark_plan_selection.py" --exe "$out/provenance/spmv_hybrid_validation" --index "$out/index.csv" --matrix-root /staff/chengjiajun/workspace/database/suitsparse --output "$out" --gpu-index 2 --selection auto_tuned_mixed --compare-auto-model --compare-vector-model --compare-full-tune --counterbalance --warm-setup-anchor --alias-identical-models --alias-identical-graphs --internal-graph-events >> "$out/runner.log" 2>&1
 rc=$?
 [ "$rc" -eq 75 ] || break
done
if [ "$rc" -eq 0 ]; then
 python3 "$out/provenance/analyze_plan_selection.py" "$out" > "$out/summary.log" 2>&1
 python3 "$out/provenance/analyze_selection_oracle.py" "$out" > "$out/oracle-summary.log" 2>&1
fi
printf 'runner exit code: %s\n' "$rc" >> "$out/runner.log"
exit "$rc"
