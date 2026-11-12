#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/selection-model/medium-row-selector-v1
while true; do
    python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index 3 "$out/provenance/spmv_plan_test" >> "$out/correctness.log" 2>&1
    rc=$?
    [ "$rc" -eq 75 ] || break
done
printf 'correctness exit code: %s\n' "$rc" >> "$out/correctness.log"
[ "$rc" -eq 0 ] || exit "$rc"
while true; do
    python3 "$out/provenance/benchmark_plan_selection.py" --exe "$out/provenance/spmv_hybrid_validation" \
        --index "$out/index.csv" --matrix-root /staff/chengjiajun/workspace/database/suitsparse \
        --output "$out" --gpu-index 3 --selection auto_medium_model --compare-vector-model-v2 \
        --counterbalance --internal-graph-events --log-model-predictions >> "$out/runner.log" 2>&1
    rc=$?
    [ "$rc" -eq 75 ] || break
done
python3 "$out/provenance/analyze_medium_plan_comparison.py" "$out" --splits "$out/provenance/replay.csv" > "$out/summary.log" 2>&1
printf 'runner exit code: %s\n' "$rc" >> "$out/runner.log"
exit "$rc"
