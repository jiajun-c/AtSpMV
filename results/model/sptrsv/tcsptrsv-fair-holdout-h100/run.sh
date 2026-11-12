#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/sptrsv/tcsptrsv-fair-holdout-h100
while true; do
 python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index 3 python3 "$out/provenance/benchmark_sptrsv_tc.py" --manifest "$out/manifest.json" --factors results/model/sptrsv/ilu-selector-holdout-h100/factors --output "$out/cases" --plan "$out/provenance/plan" --tc "$out/provenance/tcsptrsv" --tc-workdir /tmp/atspmv-tcsptrsv/TC-SpTRSV/TCSpTRSV/script --resume >> "$out/runner.log" 2>&1
 status=$?
 if [ "$status" -ne 75 ]; then break; fi
done
printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
exit "$status"
