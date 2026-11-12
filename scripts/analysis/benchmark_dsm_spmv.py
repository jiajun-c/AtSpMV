#!/usr/bin/env python3
"""Compare occupancy-limited and one-resident-CTA-per-SM SpMV schedules."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[202610281, 202610282, 202610283])
    parser.add_argument("--warp-sweep", action="store_true")
    parser.add_argument("matrices", type=Path, nargs="+")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    metadata = {
        "sources": {p: sha(root / p) for p in ["src/dsmSpmv.cu", "include/dsmSpmv.h", "bench/dsm_spmv_bench.cu"]},
        "executable": str(args.exe.resolve()), "executable_sha256": sha(args.exe),
        "inputs": {str(p.resolve()): sha(p) for p in args.matrices}, "seeds": args.seeds,
        "warp_sweep": args.warp_sweep,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_state_before": subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used", "--format=csv"], text=True),
        "timing": "Warm graph of 64 independent y=A*x calls, each restaging X; includes clearing/staging/synchronization. 11 measured interleaved batches, 2 warmup batches. Packing and plan setup excluded.",
        "selection": "Best measured configuration per kind (six shapes, two balancing modes, C=1/2/4/8); not an online selector",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = []
    for matrix in args.matrices:
        for seed in args.seeds:
            path = args.output / f"{matrix.stem}-{seed}.csv"
            with path.open("w") as output:
                subprocess.run([str(args.exe.resolve()), str(matrix.resolve()), str(seed)] + (["--warp-sweep"] if args.warp_sweep else []), stdout=output, check=True)
            with path.open() as output:
                rows = list(csv.DictReader(output))
            kinds = sorted({r["kind"] for r in rows})
            best = {kind: min((r for r in rows if r["kind"] == kind), key=lambda r: float(r["median_us"])) for kind in kinds}
            for kind, row in best.items():
                row["logical_x_staging_bytes"] = (int(row["blocks"]) // int(row["cluster"]) * int(row["cols"]) * 8) if kind in {"dsm", "one_cta_dsm"} else 0
            record = {"matrix": matrix.stem, "seed": seed, "best": best}
            if "one_cta_dsm" in best:
                dsm = best["one_cta_dsm"]
                record["matched_one_cta_global"] = next(r for r in rows if r["kind"] == "one_cta_global" and all(r[k] == dsm[k] for k in ["m", "balanced", "cluster", "blocks", "warps"]))
            summary.append(record)
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(matrix.stem, seed, {k: r["median_us"] for k, r in best.items()}, flush=True)


if __name__ == "__main__":
    main()
