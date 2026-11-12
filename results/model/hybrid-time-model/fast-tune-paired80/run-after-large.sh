#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/hybrid-time-model/fast-tune-paired80
for dep in '1015097:atspmv-counterbalanced-master.sh' '1094282:atspmv-large-suite-wrapper.sh'; do
  pid=${dep%%:*}
  pattern=${dep#*:}
  while [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | rg -q "$pattern"; do sleep 30; done
done
while true; do
  python3 "$out/provenance/benchmark_plan_selection.py" --exe "$out/provenance/spmv_hybrid_validation" --index "$out/index.csv" --matrix-root /staff/chengjiajun/workspace/database/suitsparse --output "$out" --gpu-index 3 --monitor-resident --selection auto_fast_tuned --compare-full-tune --counterbalance --extended-baselines >> "$out/runner.log" 2>&1
  rc=$?
  if [ "$rc" -ne 75 ]; then break; fi
done
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
python3 "$out/provenance/audit_plan_sweep.py" "$out" || exit $?
python3 "$out/provenance/audit_tuned_choice.py" "$out" || exit $?
python3 "$out/provenance/collect_plan_tuning.py" "$out" --output "$out/tuning.csv" || exit $?
python3 "$out/provenance/analyze_plan_selection.py" "$out" || exit $?
