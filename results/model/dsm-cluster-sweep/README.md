# H100 cluster-size DSM sweep

`final/` records six measured runs for sizes 2/4/8/16. Size 16 is nonportable and was opted in and checked for each tested kernel. Remote rank 1 and the last rank are sampled; physical SM IDs are summarized only as distinct-SM counts. `dsm_async_fixed16` changes threads per CTA as explained in the project documentation.

`validation/` contains direct/graph correctness checks, H100 and Ada CTest logs, sanitizer output and the compiled SASS. `quick.csv` contains instrumented quick checks, not performance evidence. The previous cluster-2-only dataset remains in `results/model/dsm-latency/`.
