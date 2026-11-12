# Exploratory crossover profiling

Source cases and configurations are the independently repeated 38-matrix experiment (`expanded/final/summary.json`); these are not the ongoing full-suite results. Each NCU invocation selects the second direct invocation of either the fused kernel or ordinary AtBSR compute kernel (the latter excludes any separate output-clear kernel), uses cache-control none and clock-control none, and collects SpeedOfLight, Occupancy, SchedulerStats, WarpStateStats and MemoryWorkloadAnalysis. NCU replay can substantially change cache state, and the kernels are short; NCU timings are not the event timings used for the speedup comparison. Concurrent external GPU traffic was not monitored continuously for these exploratory profiles. Use these counters as hypotheses, not definitive attribution.

Using SciPy Matrix Market loading, threshold 1024 partitions:

| case | rows | nnz | max row | long rows | long nnz / all |
| --- | ---: | ---: | ---: | ---: | ---: |
| a0nsdsil | 80016 | 355034 | 5003 | 8 | 11.3% |
| lowThrust_6 | 16928 | 207349 | 6582 | 2 | 6.3% |
| c-53 | 30235 | 372213 | 4468 | 43 | 20.6% |
| rajat20 | 86916 | 605045 | 29105 | 3 | 6.1% |

`c-53` has 91 additional rows of length 128–1023 accounting for approximately 10.7% of its nnz. The original hybrid only splits at 1024/4096, leaving these rows in unbalanced AtBSR; one warp or subwarp works a longer serial loop. `rajat20` has a single 29105-entry maximum row; dividing it into four or eight cluster CTAs may expose a long tail even though it has few long rows. These are candidate mechanisms and require controlled ablation, not established causal attribution.

Threshold search under `../threshold-search/` used 72 lower-threshold (128/256/512) group/cluster/tile configurations plus three fixed AtBSR configs and cuSPARSE. One exploratory run found c-53 improved from the previous roughly 13 us hybrid to roughly 8 us with threshold 128, C=4, 256 threads, 4 rows per cluster and tile M=8, but AtBSR was around 5.5 us. The same threshold reduction made rajat20 slower than its previous hybrid. Since GPU occupation by other users was observed near the search and no continuous contention trace was saved, confirm with isolated randomized repeats before using these as results or enabling any default tuning candidate.
