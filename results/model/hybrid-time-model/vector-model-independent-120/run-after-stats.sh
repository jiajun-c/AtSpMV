#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/hybrid-time-model/vector-model-independent-120
previous=results/model/hybrid-time-model/shared-tile-stats-paired80
pid=$(cat "$previous/queue.pid")
while [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | rg -q 'shared-tile-stats-paired80/run-after-short-cache.sh'; do sleep 30; done
while true; do
    python3 "$out/provenance/benchmark_plan_selection.py" --exe "$out/provenance/spmv_hybrid_validation" --index "$out/index.csv" --matrix-root /staff/chengjiajun/workspace/database/suitsparse --output "$out" --gpu-index 3 --monitor-resident --selection auto_model --counterbalance --extended-baselines --vector-candidates >> "$out/runner.log" 2>&1
    rc=$?
    if [ "$rc" -ne 75 ]; then break; fi
done
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
python3 "$out/provenance/audit_plan_sweep.py" "$out" || exit $?
python3 "$out/provenance/analyze_plan_selection.py" "$out" || exit $?
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 "$out/provenance/evaluate_frozen_vector_model.py" "$out" || exit $?
