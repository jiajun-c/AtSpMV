#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/sptrsv/plan-csr-independent120-h100
while true; do
 python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index 3 python3 "$out/provenance/benchmark_sptrsv_plan_pair.py" --manifest "$out/manifest.json" --factors results/model/sptrsv/csr-independent120-h100/factors --output "$out/cases" --fixed "$out/provenance/fixed" --adaptive "$out/provenance/adaptive" --resume >> "$out/runner.log" 2>&1
 status=$?
 if [ "$status" -ne 75 ]; then break; fi
done
python3 "$out/provenance/analyze_sptrsv_plan_pair.py" "$out" --allow-backend-change >> "$out/runner.log" 2>&1
printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
exit "$status"
