#!/usr/bin/env python3
"""Check paired v1/v2 statistics and incomplete-run accounting."""
import csv
import json
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    d = Path(directory)
    (d/'status').mkdir(); (d/'raw').mkdir()
    (d/'manifest.json').write_text(json.dumps([{'dataset':'known','nnz':'10'}, {'dataset':'waiting','nnz':'20'}]))
    (d/'metadata.json').write_text(json.dumps({'compare_vector_model_v2':True, 'trials':2,'log_model_predictions':True,'vector_candidates':True}))
    (d/'configs.csv').write_text('atbsr,0\nauto_vector_model,0\nauto_vector_model_v2,0\n'+''.join(f'csr_vec{w},0\n' for w in (1,2,4,8,16,32)))
    (d/'status/known.json').write_text('{"status":"ok"}')
    with (d/'raw/known.log').open('w') as f:
        for order in ('baseline_first','selection_first'):
            f.write(f'BENCH_ORDER,{order}\nPLAN_PREP,auto_vector_model,100\nPLAN_PREP,auto_vector_model_v2,200\n')
            f.write('MODEL_REFERENCE,auto_vector_model,2,1\nMODEL_PREDICTION,auto_vector_model,csr_vector,-1,-1,8,128,1,1024,2,0.8,1.6\n')
            f.write('MODEL_REFERENCE,auto_vector_model_v2,2,1\nMODEL_PREDICTION,auto_vector_model_v2,csr_vector,-1,-1,8,128,1,1024,4,0.4,0.8\n')
            for label, chosen in (('auto_vector_model',2),('auto_vector_model_v2',4)):
                for width in (1,2,4,8,16,32):
                    if width != chosen: f.write(f'MODEL_PREDICTION,{label},csr_vector,-1,-1,8,128,1,1024,{width},2,4\n')
    with (d/'training.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset','split']); writer.writeheader()
        writer.writerow({'dataset':'known','split':'holdout'})
    with (d/'raw/known.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['order','backend','us','config']); writer.writeheader()
        for order in ('baseline_first', 'selection_first'):
            for label, time in (('atbsr',3), ('cusparse',4), ('auto_vector_model',2), ('auto_vector_model_v2',1)):
                for _ in range(2):
                    config = ('csr_vector/-1/-1/8/128/1/1024/vw=2' if label=='auto_vector_model' else
                              'csr_vector/-1/-1/8/128/1/1024/vw=4' if label=='auto_vector_model_v2' else label)
                    writer.writerow({'order':order,'backend':label,'us':time,'config':config})
            for width in (1,2,4,8,16,32):
                for _ in range(2):
                    writer.writerow({'order':order,'backend':f'csr_vec{width}','us':1 if width==4 else 4,'config':str(width)})
    cmd = ['python3', str(root/'scripts/analysis/analyze_vector_v2_paired.py'), str(d), '--training-dataset', str(d/'training.csv')]
    subprocess.run(cmd, check=True, capture_output=True)
    result = json.loads((d/'v2-paired-analysis.json').read_text())
    assert result['status_counts'] == {'ok':1, 'pending':1}
    assert result['holdout']['v2_speedup_vs_v1'] == 2
    assert result['holdout']['v2_over_observed_best'] == 1
    assert result['holdout']['v1_within_5pct_best'] == 0
    assert result['holdout']['geomean_v2_setup_over_v1'] == 2
    assert abs(result['holdout']['total_cost_speedups']['1'] - 102/201) < 1e-12
    assert abs(result['holdout']['v2_prediction_ratio_mape']['mean'] - .2) < 1e-12
    assert abs(result['holdout']['v2_prediction_absolute_mape']['mean'] - .2) < 1e-12
    assert result['segments']['nnz_bucket']['<10k']['cases'] == 1
    assert result['segments']['v2_reference_calibrated']['1']['cases'] == 1
    assert result['holdout']['v2_width_within_5pct_best'] == 1
print('paired v2 analysis passed')
