#!/usr/bin/env python3
"""Plot grid-size effects with exactly balanced dense row assignments."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    results = json.loads((args.directory / "summary.json").read_text())
    selected = [r for r in results if r["m"] == 7296]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    one_grid, tuned = [], []
    for r in selected:
        configs = r["configs"]
        normal128 = next(c for c in configs if c["kind"] == "one_row_per_cta" and c["threads"] == 128)
        cyclic = sorted((c for c in configs if c["kind"] == "persistent_cyclic" and c["threads"] == 128), key=lambda c: c["blocks"])
        assert all(c["min_rows_per_cta"] == c["max_rows_per_cta"] for c in cyclic)
        axes[0].plot([c["blocks"] for c in cyclic], [c["median_us"] / normal128["median_us"] for c in cyclic], marker="o", label=f'N={r["n"]}')
        baseline = r["best_by_kind"]["one_row_per_cta"]["median_us"]
        one_grid.append(min(c["median_us"] for c in r["one_grid_cta_per_sm"]) / baseline)
        tuned.append(min(c["median_us"] for k, c in r["best_by_kind"].items() if k.startswith("persistent")) / baseline)
    axes[0].set(xlabel="Persistent CTA count", ylabel="Time / ordinary time (lower is better)", title="Same 128 threads/CTA; M=7296")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([114, 228, 456, 912, 1824], [114, 228, 456, 912, 1824])
    axes[0].legend()
    x = np.arange(len(selected))
    axes[1].bar(x-.18, one_grid, .36, label="114 CTAs, best thread count")
    axes[1].bar(x+.18, tuned, .36, label="Best persistent grid + threads")
    axes[1].set_xticks(x, [str(r["n"]) for r in selected])
    axes[1].set(xlabel="Columns N", ylabel="Time / best ordinary time", title="Tuning recovers most of the gap")
    axes[1].legend()
    for ax in axes:
        ax.axhline(1, linestyle="--", color="gray", linewidth=1)
        ax.grid(axis="y", alpha=.15)
    fig.savefig(args.directory / "grid-size-effect.png", dpi=180)
    fig.savefig(args.directory / "grid-size-effect.svg")
    m, blocks = 7296, 114
    assignment = {
        "m": m, "blocks": blocks, "normal": "CTA b owns row b; normal grid has M CTAs",
        "cyclic": [{"cta": b, "rows": list(range(b, m, blocks))} for b in range(blocks)],
        "contiguous": [{"cta": b, "rows": list(range(m*b//blocks, m*(b+1)//blocks))} for b in range(blocks)],
    }
    for layout in ["cyclic", "contiguous"]:
        assert sorted(row for cta in assignment[layout] for row in cta["rows"]) == list(range(m))
        assert all(len(cta["rows"]) == 64 for cta in assignment[layout])
    (args.directory / "row-assignment-m7296-b114.json").write_text(json.dumps(assignment, indent=2) + "\n")


if __name__ == "__main__":
    main()
