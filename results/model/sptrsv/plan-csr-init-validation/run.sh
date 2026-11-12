#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
while tmux has-session -t atsptrsv_plancsr160 2>/dev/null || tmux has-session -t atsptrsv_tccsr160 2>/dev/null; do sleep 15; done
out=results/model/sptrsv/plan-csr-init-validation
while true; do
 python3 scripts/analysis/run_on_idle_gpu.py --gpu-index 3 bash "$out/check.sh" >> "$out/runner.log" 2>&1
 status=$?
 if [ "$status" -ne 75 ]; then break; fi
done
printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
exit "$status"
