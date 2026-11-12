#!/usr/bin/env python3
"""Build a CSR-register-cache-disabled control without changing production sources.

Uses an already built Unix Makefiles CMake tree and its benchmark object/libraries.
The only source change forces the existing uncached dispatch for both DSM/global.
This removes CSR/address caching AND fixed-row unrolling, not just register storage.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    build, output = args.build.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    original = (root / "src/dsmPageRank.cu").read_text()
    marker = "int rows=(n+C*(Threads/4)-1)/(C*(Threads/4));"
    assert original.count(marker) == 1, "dispatch changed: review the ablation"
    modified = original.replace(marker, "int rows=9; // Ablation: always select Rows=0.")
    source = output / "dsmPageRank-no-csr-cache.cu"
    source.write_text(modified)
    cache = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if "=" in line and not line.startswith(("#", "//")):
            key, value = line.split("=", 1)
            cache[key.split(":")[0]] = value
    flags = {}
    for line in (build / "CMakeFiles/atspmv_dsm.dir/flags.make").read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            flags[key] = shlex.split(value)
    obj = output / "dsmPageRank-no-csr-cache.o"
    compile_command = [cache["CMAKE_CUDA_COMPILER"],
                       "-ccbin=" + cache["CMAKE_CUDA_HOST_COMPILER"],
                       *flags["CUDA_DEFINES"], *flags["CUDA_INCLUDES"],
                       *flags["CUDA_FLAGS"], "-c", str(source), "-o", str(obj)]
    link_command = shlex.split((build / "CMakeFiles/dsm_pagerank_bench.dir/link.txt").read_text())
    executable = output / "dsm_pagerank_bench_no_csr_cache"
    link_command[link_command.index("-o") + 1] = str(executable)
    link_command[link_command.index("libatspmv_dsm.a")] = str(obj)
    subprocess.run(compile_command, cwd=build, check=True)
    subprocess.run(link_command, cwd=build, check=True)
    sha = lambda value: hashlib.sha256(value).hexdigest()
    metadata = {
        "ablation": "Force uncached fallback for both DSM and global; X storage unchanged",
        "coupled_changes": ["CSR register caching", "DSM address precomputation", "fixed-row unrolling"],
        "original_source_sha256": sha(original.encode()),
        "modified_source_sha256": sha(modified.encode()),
        "executable_sha256": sha(executable.read_bytes()),
        "compile": compile_command, "link": link_command, "working_directory": str(build),
    }
    (output / "ablation-build.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(executable)


if __name__ == "__main__":
    main()
