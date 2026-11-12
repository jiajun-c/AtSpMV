#!/usr/bin/env bash
# Batch benchmark runner; each matrix lives at ROOT/name/name.mtx.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 DATA_DIR [OUTPUT_DIR]" >&2
    exit 2
fi
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DATA_DIR=$1
OUTPUT_DIR=${2:-$PROJECT_ROOT/artifacts/runs}
BENCHMARK=${BENCHMARK:-$PROJECT_ROOT/build/spmv_double}
SIZE_THRESHOLD_MB=${SIZE_THRESHOLD_MB:-10000}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-240}
[[ -d "$DATA_DIR" ]] || { echo "Missing data directory: $DATA_DIR" >&2; exit 1; }
[[ -x "$BENCHMARK" ]] || { echo "Missing executable: $BENCHMARK" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
for folder in "$DATA_DIR"/*/; do
    [[ -d "$folder" ]] || continue
    name=$(basename "$folder")
    matrix="$folder$name.mtx"
    [[ -f "$matrix" ]] || continue
    bytes=$(wc -c < "$matrix")
    if (( bytes / 1024 / 1024 > SIZE_THRESHOLD_MB )); then
        echo "Skipping oversized matrix: $name" >&2
        continue
    fi
    echo "Running: $name" >&2
    if timeout "$TIMEOUT_SECONDS" "$BENCHMARK" "$matrix" > "$OUTPUT_DIR/$name.log" 2>&1; then
        echo "Completed: $name" >&2
    else
        status=$?
        echo "Failed: $name (exit $status); see $OUTPUT_DIR/$name.log" >&2
    fi
done
