#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/hybrid-time-model/cusparse-buffer-probe
for dep in '1015097:atspmv-counterbalanced-master.sh' '1094282:atspmv-large-suite-wrapper.sh' '1140016:fast-tune-paired80/run-after-large.sh'; do
    pid=${dep%%:*}; pattern=${dep#*:}
    while [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | rg -q "$pattern"; do sleep 30; done
done
for name in sgpf5y6 nemeth20 smallworld TSOPF_RS_b162_c4; do
    matrix="/staff/chengjiajun/workspace/database/suitsparse/$name/$name.mtx"
    for seed in 20261028 20261029; do
        target="$out/h100-$name-$seed.csv"
        if [ -s "$target" ]; then continue; fi
        while true; do
            python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index 3 bash -c 'exec "$1" "$2" "$3" 31 > "$4"' _ "$out/provenance/cusparse_buffer_probe" "$matrix" "$seed" "$target.tmp"
            rc=$?
            if [ "$rc" -eq 75 ]; then rm -f "$target.tmp"; continue; fi
            if [ "$rc" -ne 0 ]; then rm -f "$target.tmp"; exit "$rc"; fi
            mv "$target.tmp" "$target"
            break
        done
    done
done
python3 "$out/provenance/analyze_cusparse_buffers.py" "$out"/h100-*.csv --output "$out/h100-analysis.json"
