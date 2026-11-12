# DSM and atomic cycle measurements

`final/` is the reproducible six-run dataset, using the final binary and source hashes recorded in `final/metadata.json`. The CSV contains per-run medians and per-trial percentile ranges, and `summary.json` aggregates the three seeds separately for 1024/4096 loop lengths.

`validation/` has correctness/sanitizer logs, the complete compiled SASS, and Nsight Compute samples. The L2 read sector counters are 6080 hits/20 misses for a separate sampled kernel; the single-CTA global atomic sample reports 4096 hits/0 misses. Nsight instrumentation and its cache state are not used for the cycle or event timings. Its aggregate Memory Workload Analysis L2 hit rate includes all traffic, including output stores, and differs from the read-only sector ratio.

`screen*.csv` files are preliminary probes using earlier benchmark revisions. They are not final performance evidence and do not have full per-run provenance.

Clock samples for returnless atomic and one-iteration atomic do not guarantee operation completion; event times include the whole kernel. The DSM async-round clock starts after cluster initialization, includes the transaction wait and rank-0 reduction, and may precede completion of the final global output store. See the project documentation for the full methodology and limitations.
