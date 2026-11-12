#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/selection-model/h100-vector-graph-train-dev
previous=results/model/selection-model/h100-vector-graph-663
while tmux has-session -t atspmv_graph_663 2>/dev/null; do sleep 30; done
while tmux has-session -t atspmv_graph_summary 2>/dev/null; do sleep 10; done
if ! grep -q '^runner exit code: 0$' "$previous/runner.log" 2>/dev/null || [ ! -f "$previous/graph-comparison.json" ]; then
    printf 'Previous Graph comparison did not complete successfully.\n' >> "$out/runner.log"
    exit 1
fi
if ! python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); sys.exit(not (x["total"] == 663 and sum(x["status_counts"].values()) == 663 and x["status_counts"].get("pending", 0) == 0))' "$previous/graph-comparison.json"; then
    printf 'Previous analysis is incomplete; refusing to combine results.\n' >> "$out/runner.log"
    exit 1
fi
while true; do
    python3 "$out/provenance/benchmark_plan_selection.py" \
        --exe "$out/provenance/spmv_hybrid_validation" --index "$out/index.csv" \
        --matrix-root /staff/chengjiajun/workspace/database/suitsparse --output "$out" \
        --gpu-index 3 --selection auto_vector_model --compare-auto-model \
        --compare-vector-model-v2 --compare-vector-model-graph --vector-candidates \
        --counterbalance --warm-setup-anchor --alias-identical-models \
        --alias-identical-graphs --internal-graph-events --log-model-predictions >> "$out/runner.log" 2>&1
    rc=$?
    [ "$rc" -eq 75 ] || break
done
printf 'runner exit code: %s\n' "$rc" >> "$out/runner.log"
if [ "$rc" -eq 0 ]; then
    python3 "$out/provenance/analyze_vector_graph_comparison.py" "$out" > "$out/analysis.log" 2>&1 || exit 1
    python3 "$out/provenance/combine_vector_graph_comparison.py" > "$out/combine.log" 2>&1 || exit 1
fi
exit "$rc"
