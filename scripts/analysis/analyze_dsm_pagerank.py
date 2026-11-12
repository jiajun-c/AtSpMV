#!/usr/bin/env python3
"""Describe a fixed PageRank sweep; correlations are not dispatch rules."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits


def features(item):
    # Keep COO entries, including duplicates and explicit zero-valued edges:
    # csrReader and the unweighted PageRank benchmark also retain these.
    with threadpool_limits(limits=4):
        a = mmread(item["path"]).tocoo()
    n = a.shape[0]
    outgoing = np.bincount(a.row, minlength=n)
    incoming = np.bincount(a.col, minlength=n)
    # PageRank uses transposed CSR: a.col is the destination/CSR row.
    nnz_by_cta = np.bincount(a.col % 8, minlength=8)
    cached = n <= 4096
    return {
        "matrix": item["matrix"], "n": n, "nnz": a.nnz,
        "mean_in_degree": a.nnz / n,
        "max_in_degree": int(incoming.max()),
        "p95_in_degree": float(np.quantile(incoming, .95)),
        "in_degree_cv": float(incoming.std() / max(incoming.mean(), 1e-30)),
        "dangling_fraction": float(np.mean(outgoing == 0)),
        "remote_fraction_c8": float(np.mean(a.row % 8 != a.col % 8)),
        "cta_nnz_max_over_mean": float(nnz_by_cta.max() / max(nnz_by_cta.mean(), 1e-30)),
        "register_cached_edges_fraction": float(np.minimum(incoming, 32).sum() / max(a.nnz, 1)) if cached else 0.,
        "cached_kernel": cached,
        "x_bytes": n * 8,
        "matrix_csr_bytes": a.nnz * 12 + (n + 1) * 4,
        "x_double_buffer_shared_per_cta": (2 * ((n + 7) // 8) + 8) * 8,
    }


def aggregate(feature, runs):
    result = dict(feature)
    for key in ["dsm_c8_us", "global_c8_us", "best_global_us",
                "local_shared_c1_us", "best_fused_graph_us", "cusparse_graph_us"]:
        values = [r[key] for r in runs if r[key] is not None]
        result[key] = float(np.median(values)) if values else None
    comparisons = {
        "matched_global": lambda r: r["global_c8_us"],
        "fused_graph": lambda r: r["best_fused_graph_us"],
        "best_non_dsm": lambda r: min(r[k] for k in [
            "best_global_us", "local_shared_c1_us", "best_fused_graph_us",
            "cusparse_graph_us"] if r[k] is not None),
    }
    for label, baseline in comparisons.items():
        ratios = [baseline(r) / r["dsm_c8_us"] for r in runs]
        result[f"speedup_{label}"] = float(np.median(ratios))
        result[f"min_repeat_speedup_{label}"] = min(ratios)
        result[f"max_repeat_speedup_{label}"] = max(ratios)
    result["repeats"] = len(runs)
    return result


def profile_summary(directory):
    keep = ["L2 Hit Rate", "L1/TEX Hit Rate", "Registers Per Thread",
            "Local Memory Spilling Requests", "Achieved Occupancy", "Waves Per SM",
            "DRAM Throughput", "Warp Cycles Per Issued Instruction"]
    result = {}
    for path in sorted(directory.glob("*.csv")):
        text = path.read_text()
        rows = list(csv.DictReader(text[text.index('"ID",'):].splitlines()))
        result[path.stem] = {}
        for label, pattern in [("global", "pageRank<8, 0,"), ("dsm", "pageRank<8, 1,")]:
            selected = [r for r in rows if pattern in r.get("Kernel Name", "")]
            assert len({r["ID"] for r in selected}) == 2, (path, label)
            metrics = {}
            for metric in keep:
                values = [float(r["Metric Value"].replace(",", "")) for r in selected
                          if r["Metric Name"] == metric and r["Metric Value"] != "no data"]
                if values:
                    metrics[metric] = float(np.median(values))
            result[path.stem][label] = metrics
    if result:
        (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--features-only", action="store_true")
    p.add_argument("--ablation-directory", type=Path)
    args = p.parse_args()
    selection = json.loads((args.directory / "selection.json").read_text())
    feature_path = args.directory / "features.json"
    if feature_path.exists():
        structural = json.loads(feature_path.read_text())
    else:
        structural = []
        for item in selection["matrices"]:
            structural.append(features(item))
            print("features:", item["matrix"], flush=True)
        feature_path.write_text(json.dumps(structural, indent=2) + "\n")
    if args.features_only:
        return
    runs = json.loads((args.directory / "summary.json").read_text())
    result = []
    max_error, candidate_count = 0., 0
    for feature in structural:
        subset = [r for r in runs if r["matrix"] == feature["matrix"]]
        assert len(subset) == len(selection["seeds"]), feature["matrix"]
        assert {r["seed"] for r in subset} == set(selection["seeds"])
        assert all(r["iterations"] == selection["iterations"] for r in subset)
        # Guard against differences in symmetry expansion between readers.
        for r in subset:
            path = args.directory / f'{r["matrix"]}-i{r["iterations"]}-s{r["seed"]}.csv'
            with path.open() as f:
                raw = list(csv.DictReader(f))
            assert all(int(x["nnz"]) == feature["nnz"] and int(x["n"]) == feature["n"] for x in raw)
            errors = [float(x["max_abs_error"]) for x in raw]
            assert all(np.isfinite(x) and x <= 1e-11 for x in errors)
            max_error = max(max_error, *errors)
            candidate_count += len(raw)
        result.append(aggregate(feature, subset))
    with (args.directory / "analysis.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)

    def stats(rows):
        return {"count": len(rows), **{
            label: {
                "median_speedup": float(np.median([r[f"speedup_{label}"] for r in rows])),
                "geomean_speedup": float(np.exp(np.mean(np.log([r[f"speedup_{label}"] for r in rows])))),
                "wins_gt_1": sum(r[f"speedup_{label}"] > 1 for r in rows),
                "all_repeats_gt_1_05": sum(r[f"min_repeat_speedup_{label}"] > 1.05 for r in rows),
            } for label in ["matched_global", "fused_graph", "best_non_dsm"]}}

    report = {"overall": stats(result), "groups": {}, "spearman_correlations": {},
              "validation": {"runs": len(runs), "candidate_rows": candidate_count,
                             "max_abs_error": max_error, "structure_counts_match": True}}
    groups = {
        "n_le_512": [r for r in result if r["n"] <= 512],
        "n_513_to_4096": [r for r in result if 512 < r["n"] <= 4096],
        "n_gt_4096": [r for r in result if r["n"] > 4096],
        "dangling": [r for r in result if r["dangling_fraction"] > 0],
        "no_dangling": [r for r in result if r["dangling_fraction"] == 0],
    }
    report["groups"] = {k: stats(v) for k, v in groups.items() if v}
    for feature in ["n", "mean_in_degree", "max_in_degree", "in_degree_cv",
                    "dangling_fraction", "remote_fraction_c8", "cta_nnz_max_over_mean",
                    "register_cached_edges_fraction"]:
        report["spearman_correlations"][feature] = {
            label: float(spearmanr([r[feature] for r in result],
                                  [r[f"speedup_{label}"] for r in result]).statistic)
            for label in ["matched_global", "best_non_dsm"]}
    report["robust_winners"] = [r["matrix"] for r in result if r["min_repeat_speedup_best_non_dsm"] > 1.05]
    (args.directory / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    profile_summary(args.directory / "profiles")
    if args.ablation_directory:
        ablation = json.loads((args.ablation_directory / "summary.json").read_text())
        base_meta = json.loads((args.directory / "metadata.json").read_text())
        variant = json.loads((args.ablation_directory / "metadata.json").read_text())["variant"]
        assert variant["original_source_sha256"] == base_meta["sources"]["src/dsmPageRank.cu"]
        comparison = []
        for name in sorted({r["matrix"] for r in ablation}):
            enabled = [r for r in runs if r["matrix"] == name]
            disabled = [r for r in ablation if r["matrix"] == name]
            assert {r["seed"] for r in enabled} == {r["seed"] for r in disabled}
            row = {"matrix": name}
            for label, subset in [("cached", enabled), ("uncached", disabled)]:
                for key in ["dsm_c8_us", "global_c8_us"]:
                    row[f"{label}_{key}"] = float(np.median([r[key] for r in subset]))
            comparison.append(row)
        (args.ablation_directory / "analysis.json").write_text(json.dumps({
            "scope": "Existing uncached fallback: CSR/address caching and fixed-row unrolling removed together",
            "comparison": comparison}, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for ax, label, title in zip(axes, ["matched_global", "best_non_dsm"],
                               ["DSM vs identical persistent global", "DSM vs fastest non-DSM control"]):
        scatter = ax.scatter([r["n"] for r in result], [r[f"speedup_{label}"] for r in result],
                             c=[r["mean_in_degree"] for r in result], cmap="viridis", s=32,
                             norm=matplotlib.colors.LogNorm(), edgecolors="black", linewidths=.3)
        ax.axhline(1, color="gray", linestyle="--", linewidth=1)
        ax.axvline(4096, color="gray", linestyle=":", linewidth=1)
        ax.set(xscale="log", yscale="log", xlabel="Nodes (log scale)", ylabel="Speedup (>1 favors DSM)", title=title)
        ax.grid(alpha=.15)
    fig.colorbar(scatter, ax=axes, label="Mean incoming degree")
    fig.savefig(args.directory / "speedups.png", dpi=180)
    fig.savefig(args.directory / "speedups.svg")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
