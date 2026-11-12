#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/sptrsv/components-independent-h100
while true; do
 python3 "$out/bench/provenance/run_on_idle_gpu.py" --gpu-index 3 python3 "$out/bench/provenance/benchmark_sptrsv_components.py" --manifest "$out/bench/manifest.json" --factors "$out/factors" --output "$out/bench/cases" --exe "$out/bench/provenance/bench" --paired --resume >> "$out/bench/runner.log" 2>&1
 status=$?
 if [ "$status" -ne 75 ]; then break; fi
done
python3 "$out/bench/provenance/analyze_sptrsv_component_pairs.py" "$out/bench" >> "$out/bench/runner.log" 2>&1
printf 'runner_exit=%s\n' "$status" >> "$out/bench/runner.log"
exit "$status"
