import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class JoinGraphResultsTest(unittest.TestCase):
    def test_join_preserves_pending_failures_and_unmeasured(self):
        script = Path(__file__).resolve().parents[1]/'scripts/analysis/join_graph_dataset_results.py'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root/'all-datasets-summary'
            summary.mkdir()
            with (summary/'all-datasets.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['dataset', 'scope'])
                writer.writeheader()
                writer.writerows([{'dataset': n, 'scope': s} for n, s in
                                  [('a', 'within_calibration_scope'), ('b', 'within_calibration_scope'),
                                   ('c', 'within_calibration_scope'), ('d', 'outside_calibration_scope')]])
            for directory, names in [('h100-vector-graph-663', ['a']),
                                     ('h100-vector-graph-train-dev', ['b', 'c'])]:
                run = root/directory
                (run/'status').mkdir(parents=True)
                (run/'manifest.json').write_text(json.dumps([{'dataset': n} for n in names]))
            run = root/'h100-vector-graph-663'
            (run/'status/a.json').write_text(json.dumps({'status': 'ok'}))
            fields = ['dataset', 'graph_speedup_vs_v2', 'graph_speedup_vs_cusparse',
                      'v2_width_ratio_mape', 'graph_width_ratio_mape']
            with (run/'graph-comparison.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(zip(fields, ['a', '1.2', '2', '.2', '.1'])))
            (root/'h100-vector-graph-train-dev/status/b.json').write_text(json.dumps({'status': 'error'}))
            subprocess.run([sys.executable, str(script), '--root', str(root)], check=True, capture_output=True)
            with (root/'h100-vector-graph-all/all-datasets-combined.csv').open(newline='') as stream:
                rows = {row['dataset']: row for row in csv.DictReader(stream)}
            self.assertEqual([rows[n]['graph_status'] for n in 'abcd'], ['ok', 'error', 'pending', 'not_measured'])
            self.assertEqual(rows['a']['graph_speedup_vs_v2'], '1.2')
            self.assertEqual(rows['b']['graph_speedup_vs_v2'], '')
            self.assertEqual(rows['c']['graph_group'], 'train_dev')


if __name__ == '__main__':
    unittest.main()
