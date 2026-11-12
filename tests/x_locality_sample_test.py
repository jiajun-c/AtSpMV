import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import mmwrite
from scipy.sparse import coo_matrix


class LocalitySampleTest(unittest.TestCase):
    def test_empty_rows_and_duplicate_entries(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "sample").mkdir()
            matrix = coo_matrix(
                (np.ones(6), ([0, 0, 0, 2, 2, 3], [1, 1, 4, 0, 7, 3])),
                shape=(4, 16),
            )
            mmwrite(base / "sample" / "sample.mtx", matrix)
            (base / "names.txt").write_text("sample\n")
            output = base / "result.json"
            subprocess.run(
                [sys.executable, str(root / "scripts/analysis/analyze_x_locality_sample.py"),
                 "--list", str(base / "names.txt"), "--matrix-root", str(base),
                 "--output", str(output)],
                check=True, stdout=subprocess.DEVNULL,
            )
            case = json.loads(output.read_text())["cases"][0]
            self.assertEqual(case["nnz_after_scipy_duplicate_coalescing"], 5)
            self.assertEqual(case["distinct_columns"], 5)
            self.assertEqual(case["distinct_64byte_x_lines"], 1)
            self.assertAlmostEqual(case["mean_nonempty_row_span"], 13 / 3)
            self.assertAlmostEqual(case["mean_within_row_sorted_column_gap"], 5)


if __name__ == "__main__":
    unittest.main()
