#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/sptrsv/capellini-pilot-h100
while true; do
 python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index 3 python3 "$out/provenance/benchmark_sptrsv_tc.py" --manifest "$out/manifest.json" --factors "$out/factors" --output "$out/cases" --plan "$out/provenance/plan" --tc "$out/provenance/capellini" --tc-workdir "$out/work" --timeout 25 --resume >> "$out/runner.log" 2>&1
 status=$?
 if [ "$status" -ne 75 ]; then break; fi
done
# Reuse the conservative result parser; in this directory "tc" names the
# locally adapted Capellini implementation, not TC-SpTRSV.
python3 "$out/provenance/analyze_sptrsv_capellini.py" "$out" --parser "$out/provenance/analyze_sptrsv_tc.py" >> "$out/runner.log" 2>&1
printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
exit "$status"
