#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/mixed-tile/all-2888-ada
while true; do
 python3 "$out/provenance/benchmark_mixed_tile.py" --exe "$out/provenance/mixed_tile_bench" --index "$out/index.csv" --matrix-root /staff/chengjiajun/workspace/database/suitsparse --output "$out" --gpu-index 1 >> "$out/runner.log" 2>&1
 rc=$?
 if [ "$rc" -ne 75 ]; then break; fi
 printf 'GPU contention; waiting and retrying\n' >> "$out/runner.log"
done
if [ "$rc" -eq 0 ]; then python3 "$out/provenance/analyze_mixed_tile.py" "$out" > "$out/summary.log" 2>&1; fi
printf 'runner exit code: %s\n' "$rc" >> "$out/runner.log"
exit "$rc"
