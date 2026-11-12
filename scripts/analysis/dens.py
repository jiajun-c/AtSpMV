from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = ROOT / "results/csv"
FIGURE_DIR = ROOT / "results/figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

import pandas as pd
df = pd.read_csv(CSV_DIR / 'density_select.csv')
data = df
data = data[data['csr.nnz'] > 1200000]
# data = data[data['csr.nrows'] < 32767]
data = data[data['csr.ncols'] < 32767]
# data = data[data['csr.stand'] > 500]
# data = data[data['csr.stand'] < 1000]

print(data[0:])