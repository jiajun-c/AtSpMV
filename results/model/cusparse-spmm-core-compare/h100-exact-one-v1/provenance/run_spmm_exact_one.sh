#!/bin/bash
set -u
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
cd "$root" || exit 1
out=results/model/cusparse-spmm-core-compare/h100-exact-one-v1
gpu=${ATSPMV_SPMM_GPU_INDEX:-3}
while true; do
    python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index "$gpu" \
        python3 "$out/provenance/benchmark_cusparse_spmm_cores.py" \
        --exe "$out/provenance/cusparse_spmm_core_compare" --out "$out/cases" --exact-one >> "$out/runner.log" 2>&1
    rc=$?
    [ "$rc" -eq 75 ] || break
done
python3 "$out/provenance/analyze_cusparse_spmm_cores.py" "$out" --exact-one > "$out/summary.log" 2>&1
printf 'benchmark exit code: %s\n' "$rc" >> "$out/runner.log"
[ "$rc" -eq 0 ] || exit "$rc"
while true; do
    python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index "$gpu" \
        python3 "$out/provenance/profile_cusparse_spmm_cores.py" \
        --exe "$out/provenance/cusparse_spmm_core_compare" --out "$out/instruction-profile" --exact-one >> "$out/profile-runner.log" 2>&1
    profile_rc=$?
    [ "$profile_rc" -eq 75 ] || break
done
printf 'profile exit code: %s\n' "$profile_rc" >> "$out/profile-runner.log"
exit "$profile_rc"
