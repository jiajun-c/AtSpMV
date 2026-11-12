#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/selection-model/h100-outside-scope-550
previous=results/model/selection-model/h100-full-2247
while tmux has-session -t atspmv_full_selection 2>/dev/null; do sleep 30; done
if ! grep -q '^runner exit code: 0$' "$previous/runner.log" 2>/dev/null; then
    printf 'Previous H100 run did not finish successfully; outside-scope run was not started.\n' >> "$out/runner.log"
    exit 1
fi
while true; do
    python3 "$out/provenance/benchmark_plan_selection.py" --exe "$out/provenance/spmv_hybrid_validation" --index "$out/index.csv" --matrix-root "$out/matrix-overlay" --output "$out" --gpu-index 2 --selection auto_model_refined --compare-full-tune --compare-mixed-tune --counterbalance --internal-graph-events --retry-shared-storage-on-oom >> "$out/runner.log" 2>&1
    rc=$?
    [ "$rc" -eq 75 ] || break
done
if [ "$rc" -eq 0 ]; then
    python3 "$out/provenance/analyze_plan_selection.py" "$out" > "$out/summary.log" 2>&1
fi
printf 'runner exit code: %s\n' "$rc" >> "$out/runner.log"
exit "$rc"
