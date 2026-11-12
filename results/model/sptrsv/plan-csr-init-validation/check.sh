#!/usr/bin/env bash
set -e
out=results/model/sptrsv/plan-csr-init-validation
"$out/sptrsv_selection_test"
"$out/sptrsv_plan_test"
