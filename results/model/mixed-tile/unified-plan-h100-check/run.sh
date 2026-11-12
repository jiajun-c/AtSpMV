#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/mixed-tile/unified-plan-h100-check
uuid=$(nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader) || exit 1
printf 'waiting for GPU 3 to become idle\n' > "$out/status.txt"
while true; do
 if pgrep -f '^python3 results/model/hybrid-time-model/plan-tune-large-v3/provenance/benchmark_plan_selection.py' > /dev/null; then sleep 30; continue; fi
 apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null) || { sleep 30; continue; }
 if printf '%s\n' "$apps" | grep -Fxq "$uuid"; then sleep 30; continue; fi
 util=$(nvidia-smi -i 3 --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null) || { sleep 30; continue; }
 if [ "$util" != 0 ]; then sleep 30; continue; fi
 sleep 30
 apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null) || continue
 if ! printf '%s\n' "$apps" | grep -Fxq "$uuid"; then break; fi
done
printf 'running correctness test\n' > "$out/status.txt"
CUDA_VISIBLE_DEVICES=3 "$out/spmv_plan_test" --quick > "$out/correctness.log" 2>&1 || { printf 'correctness failed; see correctness.log\n' > "$out/status.txt"; exit 1; }
for name in trans4 rajat09; do
 printf 'running %s\n' "$name" > "$out/status.txt"
 CUDA_VISIBLE_DEVICES=3 "$out/spmv_plan_bench" "/staff/chengjiajun/workspace/database/suitsparse/$name/$name.mtx" --fast-tune-mixed > "$out/$name.log" 2>&1 || { printf 'benchmark failed: %s\n' "$name" > "$out/status.txt"; exit 1; }
done
printf 'complete; outputs are single-run smoke measurements, not a performance study\n' > "$out/status.txt"
