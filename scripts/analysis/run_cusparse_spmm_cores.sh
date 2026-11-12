#!/bin/bash
set -u
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
cd "$root" || exit 1
out=results/model/cusparse-spmm-core-compare/h100-synthetic-v1
gpu_index=${ATSPMV_SPMM_GPU_INDEX:-3}
mkdir -p "$out/cases" "$out/provenance"
while true; do
    python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index "$gpu_index" \
        python3 "$out/provenance/benchmark_cusparse_spmm_cores.py" \
        --exe "$out/provenance/cusparse_spmm_core_compare" --out "$out/cases" >> "$out/runner.log" 2>&1
    rc=$?
    [ "$rc" -eq 75 ] || break
done
python3 "$out/provenance/analyze_cusparse_spmm_cores.py" "$out" > "$out/summary.log" 2>&1
printf 'benchmark exit code: %s\n' "$rc" >> "$out/runner.log"
[ "$rc" -eq 0 ] || exit "$rc"
for method in csr_alg2 csr_alg3 blocked_ell; do
    while true; do
        python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index "$gpu_index" \
            /usr/local/cuda-12.4/bin/ncu --csv --target-processes all --launch-count 10 \
            --metrics sm__inst_executed_pipe_tensor.sum \
            "$out/provenance/cusparse_spmm_core_compare" 4 25 32 "$method" \
            > "$out/profile-$method.csv" 2> "$out/profile-$method.log"
        profile_rc=$?
        [ "$profile_rc" -eq 75 ] || break
    done
    printf 'profile exit code: %s\n' "$profile_rc" >> "$out/profile-$method.log"
done
