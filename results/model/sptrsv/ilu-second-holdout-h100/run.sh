#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/sptrsv/ilu-second-holdout-h100
while true; do
  python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index 3 python3 "$out/provenance/benchmark_sptrsv_plan.py" --manifest "$out/manifest.json" --factors "$out/factors" --output "$out/cases" --exe "$out/provenance/bench" --resume >> "$out/runner.log" 2>&1
  status=$?
  if [ "$status" -ne 75 ]; then break; fi
done
python3 "$out/provenance/analyze_sptrsv_plan.py" "$out" >> "$out/runner.log" 2>&1
printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
exit "$status"
