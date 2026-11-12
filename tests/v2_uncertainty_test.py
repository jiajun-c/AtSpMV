import importlib.util
import math
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'scripts/analysis/analyze_v2_uncertainty.py'
spec = importlib.util.spec_from_file_location('v2_uncertainty', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class V2UncertaintyTest(unittest.TestCase):
    def test_family_weighting_and_paired_metrics(self):
        def row(name, speed, old, new):
            return {'dataset': name, 'training_split': 'holdout',
                    'v2_speedup_vs_v1': str(speed),
                    'v1_all_width_ratio_mape': str(old),
                    'v2_all_width_ratio_mape': str(new)}
        rows = [row('alpha1', 2, .2, .1), row('alpha2', 2, .2, .1),
                row('beta1', .5, .1, .2)]
        result = module.analyze(rows, 500, 42)
        self.assertEqual(result['all']['families'], 2)
        self.assertEqual(result['all']['cases'], 3)
        self.assertAlmostEqual(result['all']['paired_speedup'], math.exp(math.log(2) / 3))
        self.assertAlmostEqual(result['all']['width_ratio_mape_reduction'], .1 / 3)
        self.assertEqual(result['all'], result['holdout'])
        self.assertEqual(result, module.analyze(rows, 500, 42))


if __name__ == '__main__':
    unittest.main()
