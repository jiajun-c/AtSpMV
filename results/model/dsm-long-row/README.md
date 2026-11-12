# FP64 single-SpMV DSM long-row experiment

`final/` contains reproducible sweeps with source/binary/input hashes and three measurement seeds. `rectangular/` extends the real-matrix set with all eight locally available primary Matrix Market files meeting: 8 <= rows <= 2048, columns >= 8192, nnz/rows >= 2048, nnz <= 8,000,000. This structural screen was applied before measuring those eight matrices; it is targeted exploration, not a random holdout.

`validation/` records correctness and sanitizer checks. CUDA 12.4 sanitizer logs are unsuccessful historical checks; use the explicitly named CUDA 13.0 logs and validation.json for final validation. The full CUDA 13.0 racecheck sweep was interrupted in favor of a smaller synchronization-focused case; it is not a pass.

`probe/`, `quack-probe/`, `probe.csv`, and `lp_osa_30-probe.csv` are exploratory outputs from earlier source versions, without complete build provenance. Do not mix their numbers with final results.
