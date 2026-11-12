#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/mixed-tile/h100-fair-cap-paired80
while pgrep -f '^python3 results/model/hybrid-time-model/fast-tune-paired80/provenance/benchmark_plan_selection.py' > /dev/null; do sleep 30; done
while true; do
 python3 "$out/provenance/benchmark_mixed_tile.py" --exe "$out/provenance/mixed_tile_bench" --index "$out/index.csv" --matrix-root /staff/chengjiajun/workspace/database/suitsparse --output "$out" --gpu-index 3 >> "$out/runner.log" 2>&1
 rc=$?
 if [ "$rc" -ne 75 ]; then break; fi
done
if [ "$rc" -eq 0 ]; then python3 "$out/provenance/analyze_mixed_tile.py" "$out" > "$out/summary.log" 2>&1; fi
printf 'runner exit code: %s\n' "$rc" >> "$out/runner.log"
exit "$rc"
