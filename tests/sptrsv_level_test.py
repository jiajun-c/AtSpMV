#!/usr/bin/env python3
"""End-to-end GPU smoke tests for the experimental SpTRSV benchmark."""
import csv
import io
import os
import pathlib
import subprocess
import sys
import tempfile

exe = sys.argv[1]
with tempfile.TemporaryDirectory() as tmp:
    # Includes duplicate entries, empty strict-lower rows, a nonunit number of
    # rows per warp, varied degrees, and a chain of dependencies.
    matrix = pathlib.Path(tmp) / "triangular.mtx"
    entries = [(i + 1, j + 1) for i in range(37) for j in range(i) if (i * 17 + j * 7) % 13 < 3]
    entries += [(11, 1), (11, 1), (37, 36)]
    matrix.write_text("%%%%MatrixMarket matrix coordinate pattern general\n37 37 %d\n" % len(entries) + "".join(f"{i} {j}\n" for i, j in entries))
    result = subprocess.run([exe, str(matrix)], capture_output=True, text=True, check=True, timeout=60)
    records = list(csv.DictReader(io.StringIO(result.stdout)))
    assert {r["variant"] for r in records if not r["variant"].startswith(("push-", "tile-wait-", "shared-", "cached-"))} == {"cusparse", "cusparse-host", "packed-adaptive", "packed-sorted", "queue", "ordered-grid-1", "ordered-grid-4", "ordered-grid-16", "ordered-grid-32", "ordered-adaptive", "ordered-1", "ordered-4", "ordered-8", "ordered-16", "ordered-32", "blocked-16", "blocked-64", "blocked-256", "components", "components-warp", "components-warp-1", "components-warp-4", "components-warp-8", "components-warp-32", "components-host", "components-host-1", "components-host-4", "components-host-8", "components-host-32", "components-1", "components-4", "components-8", "components-32", "adaptive", "single-cta-1", "single-cta-4", "single-cta-32", "1", "2", "4", "8", "16", "32"}
    assert all(float(r["max_error"]) < 1e-10 and float(r["backward_error"]) < 1e-12 and float(r["solve_us"]) > 0 for r in records)
    fast = subprocess.run([exe, str(matrix)], capture_output=True, text=True, check=True, timeout=60, env={**os.environ, "ATSPTRSV_CACHED_ONLY": "1"})
    fast_records = list(csv.DictReader(io.StringIO(fast.stdout)))
    assert {r["variant"] for r in fast_records if not r["variant"].startswith("selector-")} == {"cusparse", "cusparse-host", "cached-tiles-128", "cached-tiles-256", "cached-tiles-512", "cached-tiles-host-128", "cached-tiles-host-256", "cached-tiles-host-512"}
    assert len([r for r in fast_records if r["variant"] in ("selector-cached", "selector-cusparse")]) == 1
    assert all(float(r["backward_error"]) < 1e-12 for r in fast_records)
    six = subprocess.run([exe, str(matrix)], capture_output=True, text=True, check=True, timeout=60, env={**os.environ, "ATSPTRSV_CACHED_ONLY": "1", "ATSPTRSV_TILE_K2": "1"})
    six_records = list(csv.DictReader(io.StringIO(six.stdout)))
    assert any(r["variant"] == "cached-tiles-512" for r in six_records)
    assert all(float(r["backward_error"]) < 1e-12 for r in six_records)
    diagonal = pathlib.Path(tmp) / "diagonal.mtx"
    diagonal.write_text("%%MatrixMarket matrix coordinate real general\n4 4 4\n1 1 1\n2 2 2\n3 3 3\n4 4 4\n")
    rejected = subprocess.run([exe, str(diagonal), "--factor"], capture_output=True, text=True, check=True, timeout=60, env={**os.environ, "ATSPTRSV_CACHED_ONLY": "1"})
    assert "selector-cusparse" in {r["variant"] for r in csv.DictReader(io.StringIO(rejected.stdout))}
    rect = pathlib.Path(tmp) / "rect.mtx"
    rect.write_text("%%MatrixMarket matrix coordinate pattern general\n2 3 1\n1 1\n")
    invalid = subprocess.run([exe, str(rect)], capture_output=True, text=True, timeout=60)
    assert invalid.returncode != 0 and "square" in invalid.stderr

    # Preserve real factor values, merge repeated coordinates and reject a
    # zero diagonal or entries outside the lower triangle.
    factor = pathlib.Path(tmp) / "factor.mtx"
    factor.write_text("%%MatrixMarket matrix coordinate real general\n4 4 9\n1 1 2\n2 1 -0.25\n2 1 -0.25\n2 2 3\n3 1 0.5\n3 2 -0.125\n3 3 1.5\n4 3 0.75\n4 4 4\n")
    result = subprocess.run([exe, str(factor), "--factor"], capture_output=True, text=True, check=True, timeout=60)
    assert all(float(r["max_error"]) < 1e-10 for r in csv.DictReader(io.StringIO(result.stdout)))
    factor.write_text("%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 1\n2 1 1\n")
    invalid = subprocess.run([exe, str(factor), "--factor"], capture_output=True, text=True, timeout=60)
    assert invalid.returncode != 0 and "diagonal" in invalid.stderr
    factor.write_text("%%MatrixMarket matrix coordinate real general\n2 2 3\n1 1 1\n1 2 1\n2 2 1\n")
    invalid = subprocess.run([exe, str(factor), "--factor"], capture_output=True, text=True, timeout=60)
    assert invalid.returncode != 0 and "lower triangular" in invalid.stderr

    # Exercise independent blocks, long chains, and fan-in/fan-out. These
    # patterns stress block barriers and queue publication over graph replays.
    import random
    rng = random.Random(2718)
    for kind in ("chain", "fan", "random"):
        n = 130
        coords = [(i + 1, i + 1, 1.0) for i in range(n)]
        for i in range(1, n):
            js = ([i - 1] if kind == "chain" else
                  [0] if kind == "fan" else
                  [j for j in range(i) if rng.random() < .035])
            coords += [(i + 1, j + 1, -0.4 / max(len(js), 1)) for j in js]
        factor.write_text("%%%%MatrixMarket matrix coordinate real general\n%d %d %d\n" % (n, n, len(coords)) + "".join(f"{i} {j} {v:.17g}\n" for i, j, v in coords))
        result = subprocess.run([exe, str(factor), "--factor"], capture_output=True, text=True, check=True, timeout=60)
        records = list(csv.DictReader(io.StringIO(result.stdout)))
        variants = [r["variant"] for r in records]
        assert len(variants) == len(set(variants))
        assert {"cusparse", "packed-adaptive", "cached-tiles-512", "cached-components-256", "cached-components-512"} <= set(variants)
        assert all(float(r["max_error"]) < 1e-10 and float(r["backward_error"]) < 1e-12 for r in records)
