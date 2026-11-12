#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/applications/mfem-heat/r6-linear
while true; do
 OPENBLAS_NUM_THREADS=1 python3 scripts/analysis/run_on_idle_gpu.py --gpu-index 3 python3 "$out/provenance/benchmark_mfem_heat.py" --exe /tmp/atspmv-mfem-heat-r6-frozen --mesh /tmp/atspmv-mfem-v4.8/data/star.mesh --output "$out" --refine 6 --steps 100 --alpha 0 --rounds 2 >> "$out/runner.log" 2>&1
 status=$?
 if [ "$status" -ne 75 ]; then break; fi
done
printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
exit "$status"
