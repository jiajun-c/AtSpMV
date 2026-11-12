#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/mixed-tile/h100-paired80
while true; do
    python3 "$out/provenance/benchmark_mixed_tile.py" --exe "$out/provenance/mixed_tile_bench" --index "$out/index.csv" --matrix-root /staff/chengjiajun/workspace/database/suitsparse --output "$out" --gpu-index 3 >> "$out/runner.log" 2>&1
    rc=$?
    if [ "$rc" -ne 75 ]; then break; fi
done
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
python3 "$out/provenance/analyze_mixed_tile.py" "$out" > "$out/summary.log" 2>&1
