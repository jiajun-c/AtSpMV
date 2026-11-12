#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/sptrsv/ilu-upper-transpose-h100
# Do not contend with our earlier non-unit experiment.
while tmux has-session -t atsptrsv_nonunit 2>/dev/null; do sleep 15; done
while true; do
 python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index 3 python3 "$out/provenance/benchmark_sptrsv_plan.py" --manifest "$out/manifest.json" --factors "$out/factors" --output "$out/cases" --exe "$out/provenance/plan" --resume >> "$out/runner.log" 2>&1
 status=$?
 if [ "$status" -ne 75 ]; then break; fi
done
python3 "$out/provenance/analyze_sptrsv_plan.py" "$out" >> "$out/runner.log" 2>&1
python3 "$out/provenance/analyze_sptrsv_plan.py" "$out" --forward-tolerance 1e-10 >> "$out/runner.log" 2>&1
printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
exit "$status"
