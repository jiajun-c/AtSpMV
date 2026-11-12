#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
while tmux has-session -t atspmv_mfem_large 2>/dev/null; do sleep 10; done
for depth in 2 4 8; do
 out=results/applications/mfem-heat/prefetch-pilot/depth-$depth
 mkdir -p "$out"
 while true; do
  OPENBLAS_NUM_THREADS=1 python3 scripts/analysis/run_on_idle_gpu.py --gpu-index 3 python3 scripts/analysis/benchmark_mfem_heat.py --exe /tmp/atspmv-mfem-apps/apps/mfem/mfem_heat --mesh /tmp/atspmv-mfem-v4.8/data/star.mesh --output "$out" --refine 6 --steps 20 --alpha 0 --rounds 1 --backends cusparse,atbsr --prefetch "$depth" >> "$out/runner.log" 2>&1
  status=$?
  if [ "$status" -ne 75 ]; then break; fi
 done
 printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
 if [ "$status" -ne 0 ]; then exit "$status"; fi
done
