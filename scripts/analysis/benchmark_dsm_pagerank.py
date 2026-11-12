#!/usr/bin/env python3
"""Run complete PageRank solves sequentially; do not overlap GPU measurements."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant-manifest", type=Path,
                        help="Build provenance for a deliberately modified control executable")
    parser.add_argument("--iterations", type=int, nargs="+", default=[100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261101, 20261102, 20261103])
    parser.add_argument("matrices", type=Path, nargs="+")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=True)
    names = [p.stem for p in args.matrices]
    if len(names) != len(set(names)):
        parser.error("matrix basenames must be unique")
    metadata = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "executable": str(args.exe.resolve()),
        "executable_sha256": sha256(args.exe),
        "sources": {p: sha256(root / p) for p in [
            "src/dsmPageRank.cu", "bench/dsm_pagerank_bench.cu"]},
        "inputs": {str(p.resolve()): sha256(p) for p in args.matrices},
        "iterations": args.iterations,
        "seeds": args.seeds,
        "timing": "CUDA events around one graph containing the complete solve; "
                  "initial copy/staging and final store included; setup excluded; "
                  "3 warmup and 15 randomized interleaved measurement batches",
        "gpu_state_before": subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used,driver_version",
            "--format=csv"], text=True),
    }
    if args.variant_manifest:
        metadata["variant"] = json.loads(args.variant_manifest.read_text())
        if metadata["variant"]["executable_sha256"] != metadata["executable_sha256"]:
            parser.error("variant manifest does not match the executable")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = []
    for matrix in args.matrices:
        for iterations in args.iterations:
            for seed in args.seeds:
                path = args.output / f"{matrix.stem}-i{iterations}-s{seed}.csv"
                command = [str(args.exe.resolve()), str(matrix.resolve()), str(iterations), str(seed)]
                with path.open("w") as output:
                    subprocess.run(command, stdout=output, check=True)
                with path.open() as output:
                    rows = list(csv.DictReader(output))
                def best(kind, cluster=None):
                    return min((float(r["median_us"]) for r in rows
                                if r["kind"].startswith(kind)
                                and (cluster is None or int(r["cluster"]) == cluster)),
                               default=None)
                # Fix DSM at C=8, rather than selecting its best cluster per graph.
                dsm = best("persistent_dsm", 8)
                record = {
                    "matrix": matrix.stem, "iterations": iterations, "seed": seed,
                    "dsm_c8_us": dsm,
                    "global_c8_us": best("persistent_global", 8),
                    "best_global_us": best("persistent_global"),
                    "local_shared_c1_us": best("persistent_dsm", 1),
                    "best_fused_graph_us": best("fused_csr_graph"),
                    "cusparse_graph_us": best("cusparse_graph"),
                }
                summary.append(record)
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
