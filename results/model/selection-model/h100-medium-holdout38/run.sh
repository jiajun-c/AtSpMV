#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/selection-model/h100-medium-holdout38
previous=results/model/selection-model/h100-v2-full-2247
while tmux has-session -t atspmv_v2_queued 2>/dev/null; do sleep 30; done
if ! grep -q '^runner exit code: 0$' "$previous/runner.log" 2>/dev/null; then
    printf 'Previous H100 run did not finish successfully.\n' >> "$out/runner.log"
    exit 1
fi
reference=results/model/selection-model/h100-absolute-reference-v1
while true; do
    python3 "$previous/provenance/run_on_idle_gpu.py" --gpu-index 3 "$reference/provenance/spmv_plan_test" >> "$reference/h100-correctness.log" 2>&1
    test_rc=$?
    [ "$test_rc" -eq 75 ] || break
done
printf 'absolute reference correctness exit code: %s\n' "$test_rc" >> "$reference/h100-correctness.log"
python3 "$out/provenance/benchmark_medium_unified.py" --replay "$out/provenance/replay.csv" \
    --exe "$out/provenance/spmv_hybrid_validation" --matrix-root /staff/chengjiajun/workspace/database/suitsparse \
    --output "$out" --idle-helper "$out/provenance/run_on_idle_gpu.py" \
    --expected-hashes "$out/provenance/matrix-sha256.json" --gpu-index 3 >> "$out/runner.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then
    flock "$out/analysis.lock" python3 "$out/provenance/analyze_medium_unified.py" "$out" --replay "$out/provenance/replay.csv" > "$out/summary.log" 2>&1
fi
printf 'runner exit code: %s\n' "$rc" >> "$out/runner.log"
exit "$rc"
