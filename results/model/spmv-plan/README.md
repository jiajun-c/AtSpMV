# Unified SpMV Plan validation

See validation.json for build/source hashes, scope, and example selections. Tests use H100 (physical GPU3) and RTX5880 (physical GPU0); GPU2 is not used.

spmv_plan_bench outputs a metadata line, then CSV candidate timings. The synthetic example is 64 x 65536 with 16384 nonzeros/row. Both tuned examples use the default shortlist; these are initialization demonstrations, not repeated benchmark claims.

Final checks: h100-test.log, ada-test.log, no-dsm-build-test.log, ctest.log, memcheck.log, synccheck.log, racecheck.log. The racecheck command uses --quick (smaller CSR, all kernels instrumented). racecheck-two-stage.log is an unsuccessful filtered instrumentation attempt, retained for transparency; it is not counted as passing.
