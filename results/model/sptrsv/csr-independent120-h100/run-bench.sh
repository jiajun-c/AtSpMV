#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/sptrsv/csr-independent120-h100
while true; do
 python3 "$out/candidates/provenance/run_on_idle_gpu.py" --gpu-index 3 python3 "$out/candidates/provenance/benchmark_sptrsv_threads.py" --manifest "$out/candidates/manifest.json" --factors "$out/factors" --output "$out/candidates/cases" --exe "$out/candidates/provenance/bench" --csr-pilot --resume >> "$out/candidates/runner.log" 2>&1
 status=$?
 if [ "$status" -ne 75 ]; then break; fi
done
python3 "$out/candidates/provenance/analyze_sptrsv_csr_pilot.py" "$out/candidates" >> "$out/candidates/runner.log" 2>&1
printf 'runner_exit=%s\n' "$status" >> "$out/candidates/runner.log"
exit "$status"
