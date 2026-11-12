# Validation scope

- `value-updates.log`: 84 numerical update / graph configurations, including
  16-bit fallback and compressed int32 columns with byte, ushort and wide tiles.
- `plan.log`: 956 direct / Graph / reference checks.
- `compressed-memcheck.log`: Compute Sanitizer 12.6, `--compressed-only`,
  three compressed/fallback configurations, zero memory errors.
- `value-updates-memcheck.log`: the attempted full update-suite memory check
  failed in `long_row_detail::reduceRows<4,1>` (the DSM path), with invalid shared
  writes. It is preserved as a failure, not counted as a passing check. The
  cause has not been determined. The MFEM comparison uses normal AtBSR and
  cuSPARSE and never launches this DSM kernel.

Independent CPU reference fields are in `../optimized-cpu-reference/`.
