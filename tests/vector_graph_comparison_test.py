import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class GraphComparisonTest(unittest.TestCase):
    def test_combine_keeps_groups_and_weights_each_matrix(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for directory, records in [('h100-vector-graph-663',[('a',2)]),
                                       ('h100-vector-graph-train-dev',[('b',1),('c',1)])]:
                run = base/directory;run.mkdir()
                (run/'graph-comparison.json').write_text(json.dumps({'cases':len(records),'status_counts':{'ok':len(records)}}))
                with (run/'graph-comparison.csv').open('w',newline='') as f:
                    names=['dataset','graph_speedup_vs_v2','graph_speedup_vs_cusparse','v2_width_ratio_mape','graph_width_ratio_mape']
                    w=csv.DictWriter(f,fieldnames=names);w.writeheader()
                    for name,speed in records:
                        w.writerow(dict(zip(names,[name,speed,speed,.2,.1])))
            subprocess.run([sys.executable,str(root/'scripts/analysis/combine_vector_graph_comparison.py'),
                            '--root',str(base),'--expected','3'],check=True,stdout=subprocess.DEVNULL)
            result=json.loads((base/'h100-vector-graph-all/summary.json').read_text())
            self.assertEqual(result['cases'],3)
            self.assertAlmostEqual(result['graph_speedup_vs_v2'],2 ** (1/3))
            self.assertAlmostEqual(result['graph_width_ratio_mape'],.1)

    def test_paired_times_and_width_prediction_errors(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run/'raw').mkdir()
            (run/'status').mkdir()
            (run/'manifest.json').write_text(json.dumps([{'dataset':'fixture'}]))
            (run/'metadata.json').write_text(json.dumps({'trials':1}))
            (run/'status/fixture.json').write_text(json.dumps({'status':'ok'}))
            with (run/'raw/fixture.csv').open('w',newline='') as f:
                w = csv.DictWriter(f,fieldnames=['order','backend','us']);w.writeheader()
                for order in ('baseline_first','selection_first'):
                    for name,us in [('atbsr',2),('cusparse',4),('auto_vector_model_v2',2),
                                    ('auto_vector_model_graph',1),
                                    *((f'csr_vec{width}',2) for width in (1,2,4,8,16,32))]:
                        w.writerow({'order':order,'backend':name,'us':us})
            with (run/'raw/fixture.log').open('w') as f:
                for order in ('baseline_first','selection_first'):
                    f.write(f'BENCH_ORDER,{order}\n')
                    for label,value in [('auto_vector_model_v2',1),('auto_vector_model_graph',1.1)]:
                        for width in (1,2,4,8,16,32):
                            f.write(f'MODEL_PREDICTION,{label},csr_vector,-1,-1,8,128,1,1024,{width},{value},1\n')
            subprocess.run([sys.executable,str(root/'scripts/analysis/analyze_vector_graph_comparison.py'),str(run)],
                           check=True,stdout=subprocess.DEVNULL)
            result=json.loads((run/'graph-comparison.json').read_text())
            self.assertEqual(result['cases'],1)
            self.assertAlmostEqual(result['graph_speedup_vs_v2'],2)
            self.assertAlmostEqual(result['graph_speedup_vs_cusparse'],4)
            self.assertAlmostEqual(result['v2_width_ratio_mape'],0)
            self.assertAlmostEqual(result['graph_width_ratio_mape'],.1)


if __name__ == '__main__': unittest.main()
