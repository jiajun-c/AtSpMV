#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/hybrid-time-model/shared-row-sort-h100
for dep in '1015097:atspmv-counterbalanced-master.sh' '1094282:atspmv-large-suite-wrapper.sh' '1140016:fast-tune-paired80/run-after-large.sh' '1158802:cusparse-buffer-probe/run-after-fast.sh'; do
    pid=${dep%%:*}; pattern=${dep#*:}
    while [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | rg -q "$pattern"; do sleep 30; done
done
for name in lp_sc205 a0nsdsil shermanACb; do
    matrix="/staff/chengjiajun/workspace/database/suitsparse/$name/$name.mtx"
    i=0
    for version in previous current current previous; do
        target="$out/$name-$i-$version"
        if [ -s "$target.csv" ] && [ -s "$target.log" ]; then i=$((i+1)); continue; fi
        while true; do
            python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index 3 bash -c 'exec "$1" "$2" "$3" 20261028 9 > "$4.csv.tmp" 2> "$4.log.tmp"' _ "$out/provenance/$version" "$matrix" "$out/config.csv" "$target"
            rc=$?
            if [ "$rc" -eq 75 ]; then rm -f "$target.csv.tmp" "$target.log.tmp"; continue; fi
            if [ "$rc" -ne 0 ]; then echo "failed $target rc=$rc"; exit "$rc"; fi
            mv "$target.csv.tmp" "$target.csv" && mv "$target.log.tmp" "$target.log" || exit 1
            break
        done
        i=$((i+1))
    done
done
python3 "$out/provenance/analyze_row_sort_comparison.py" "$out" --output "$out/analysis.json"
