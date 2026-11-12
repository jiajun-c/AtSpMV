#!/usr/bin/env python3
"""Measure explicit equal-length row assignments without DSM or sparse gathers."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[202610291, 202610292, 202610293])
    args = p.parse_args()
    # 7296 = 114 SMs * 64 rows: identical row counts for B=114/228/456/912/1824.
    cases = [(114, 1024), (128, 1024), (1024, 256), (1024, 4096),
             (7296, 256), (7296, 1024), (7296, 4096), (7296, 8192),
             (8192, 1024), (32768, 1024), (512, 32768), (4096, 16384)]
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    metadata = {
        "source_sha256": sha(root / "bench/gemv_persistent_bench.cu"),
        "executable": str(args.exe.resolve()), "executable_sha256": sha(args.exe),
        "cases": cases, "seeds": args.seeds,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_state_before": subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used", "--format=csv"], text=True),
        "method": "FP64 row-major dense GEMV; one-row CTA vs cyclic/contiguous persistent rows; 128/256/512/1024 threads; grids min(M,SMs*min(occupancy,1/2/4/8/full)); global X only; no artificial shared-memory reservation. 3 warmup +15 randomized batches of 8 graph nodes; report per-call event time. Small A may remain in L2; large A streams from memory. cuBLAS as a separate reference.",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    all_rows = []
    for m, n in cases:
        for seed in args.seeds:
            path = args.output / f"m{m}-n{n}-s{seed}.csv"
            with path.open("w") as f:
                subprocess.run([str(args.exe.resolve()), str(m), str(n), str(seed)], stdout=f, check=True)
            with path.open() as f:
                all_rows.extend(csv.DictReader(f))
        rows = [r for r in all_rows if int(r["m"]) == m and int(r["n"]) == n]
        print(m, n, {kind: round(min(float(r["median_us"]) for r in rows if r["kind"] == kind), 3)
                     for kind in sorted({r["kind"] for r in rows})}, flush=True)
    summaries = []
    for m, n in cases:
        rows = [r for r in all_rows if int(r["m"]) == m and int(r["n"]) == n]
        configs = {}
        for r in rows:
            key = (r["kind"], int(r["threads"]), int(r["blocks"]))
            configs.setdefault(key, []).append(r)
        values = []
        for key, group in configs.items():
            assert len(group) == len(args.seeds)
            r = group[0]
            values.append({"kind": key[0], "threads": key[1], "blocks": key[2],
                           "sm_count": int(r["sm_count"]),
                           "min_rows_per_cta": int(r["min_rows_per_cta"]),
                           "max_rows_per_cta": int(r["max_rows_per_cta"]),
                           "median_us": statistics.median(float(x["median_us"]) for x in group),
                           "max_abs_error": max(float(x["max_abs_error"]) for x in group)})
        summary = {"m": m, "n": n, "a_bytes": m*n*8, "configs": values}
        summary["best_by_kind"] = {kind: min((v for v in values if v["kind"] == kind), key=lambda v: v["median_us"])
                                   for kind in sorted({v["kind"] for v in values})}
        summary["one_grid_cta_per_sm"] = [v for v in values if v["kind"].startswith("persistent")
                                           and v["blocks"] == min(m, v["sm_count"])]
        summaries.append(summary)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
