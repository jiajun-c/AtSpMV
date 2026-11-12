#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/applications/mfem-heat/rebuild-reference-r7
while tmux has-session -t atspmv_mfem_gpu_struct 2>/dev/null; do sleep 10; done
while true; do
 OPENBLAS_NUM_THREADS=1 python3 scripts/analysis/run_on_idle_gpu.py --gpu-index 3 python3 "$out/benchmark.py" --exe /tmp/atspmv-mfem-gpu-structure --mesh /tmp/atspmv-mfem-v4.8/data/star.mesh --output "$out" --refine 7 --steps 20 --alpha 0.01 --rounds 4 --prefetch 4 --backends cusparse,mfem-rebuild,mfem,atbsr --gpu-assembly >> "$out/runner.log" 2>&1
 status=$?
 if [ "$status" -ne 75 ]; then break; fi
done
printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
exit "$status"
