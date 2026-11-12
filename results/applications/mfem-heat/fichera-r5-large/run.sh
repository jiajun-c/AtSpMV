#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/applications/mfem-heat/fichera-r5-large
while tmux has-session -t atspmv_mfem_next 2>/dev/null; do sleep 10; done
while true; do
 OPENBLAS_NUM_THREADS=1 python3 scripts/analysis/run_on_idle_gpu.py --gpu-index 3 python3 scripts/analysis/benchmark_mfem_heat.py --exe /tmp/atspmv-mfem-apps/apps/mfem/mfem_heat --mesh /tmp/atspmv-mfem-v4.8/data/fichera-mixed.mesh --output "$out" --refine 5 --steps 100 --alpha 0 --rounds 2 --backends cusparse,atbsr >> "$out/runner.log" 2>&1
 status=$?
 if [ "$status" -ne 75 ]; then break; fi
done
printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
exit "$status"
