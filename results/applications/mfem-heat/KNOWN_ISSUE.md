# Historical experiments before the coefficient-coherence fix

The original MFEM ex16 is a CPU example. Its raw `GridFunction::operator()(i)`
access does not update host/device validity flags. Enabling CUDA and full assembly
without an explicit `HostReadWrite()` before the coefficient loop can leave the
GPU diffusion coefficient at stale values. Backend agreement, CG convergence,
mass conservation, and energy decay do not prove the intended physical model.

Do not use runs prior to the `coherent-*` series as physical-application performance
claims. Some versions used correct bulk/device coefficient updates but compared
against this invalid baseline, and their full-field checks rejected the mismatch.
The current application synchronizes the coefficient host view in both branches;
new validation includes the upstream CPU ex16 as an independent oracle.
Previous data are retained for debugging/provenance, not as evidence of speedup.
Standalone SpMV value-update and sparse-sum tests are independent of this issue.
