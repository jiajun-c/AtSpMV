import pandas as pd
import numpy as np
import os

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = os.environ.get("ATSPMV_WORKSPACE", str(ROOT.parent))
OUT = ROOT / "results/tables"
OUT.mkdir(parents=True, exist_ok=True)

# ── Helper: clean numeric strings with spaces ──
def clean_num(s):
    if isinstance(s, str):
        return float(s.strip().replace('"', '').replace(',', ''))
    return float(s)

# ============================================================
# 1. SuiteSparse 稀疏矩阵基准测试子集
# ============================================================
density = pd.read_csv(ROOT / "results/csv/density_select.csv")

# Keep necessary columns
df1 = density[["matrix", "csr.nrows", "csr.ncols", "csr.nnz", "density"]].copy()
df1.columns = ["Matrix", "Rows", "Cols", "NNZ", "Density"]
df1["Rows"] = df1["Rows"].apply(clean_num).astype(int)
df1["Cols"] = df1["Cols"].apply(clean_num).astype(int)
df1["NNZ"] = df1["NNZ"].apply(clean_num).astype(int)
df1["Density"] = df1["Density"].apply(clean_num)
# Remove .mtx suffix from matrix name
df1["Matrix"] = df1["Matrix"].str.replace(".mtx", "", regex=False)

out1 = os.path.join(OUT, "SuiteSparse_Matrix_Subset.xlsx")
df1.to_excel(out1, index=False, sheet_name="Matrix Subset")
print(f"[1] {out1}  —  {len(df1)} matrices")

# ============================================================
# 2. A100 性能对比实测数据
# ============================================================
a100 = pd.read_csv(os.path.join(BASE, "tcspmv", "fullres.csv"))

df2 = a100[["matrix", "nrow", "ncol", "avg_nnz",
            "gflopscusp", "gflopsdasp", "gflopsamgt", "gflopsdy"]].copy()
df2.columns = ["Matrix", "Rows", "Cols", "Avg_NNZ",
               "cuSPARSE_GFlops", "DASP_GFlops", "amgT_GFlops", "AtSpMV_GFlops"]
for c in ["Rows", "Cols"]:
    df2[c] = df2[c].apply(clean_num).astype(int)
for c in ["Avg_NNZ", "cuSPARSE_GFlops", "DASP_GFlops", "amgT_GFlops", "AtSpMV_GFlops"]:
    df2[c] = df2[c].apply(clean_num)
df2["Matrix"] = df2["Matrix"].str.strip().str.replace('"', '').str.replace(".mtx", "", regex=False)

out2 = os.path.join(OUT, "AtSpMV_A100_Performance.xlsx")
df2.to_excel(out2, index=False, sheet_name="A100 Performance")
print(f"[2] {out2}  —  {len(df2)} matrices")

# ============================================================
# 3. H100 性能对比实测数据
# ============================================================
h100 = pd.read_csv(os.path.join(BASE, "tcspmvh100", "fullres_h100.csv"))

df3 = h100[["matrix", "nrow", "ncol", "avg_nnz",
            "gflopscusp", "gflopsdasp", "gflopsamgt", "gflopsdy"]].copy()
df3.columns = ["Matrix", "Rows", "Cols", "Avg_NNZ",
               "cuSPARSE_GFlops", "DASP_GFlops", "amgT_GFlops", "AtSpMV_GFlops"]
for c in ["Rows", "Cols"]:
    df3[c] = df3[c].apply(clean_num).astype(int)
for c in ["Avg_NNZ", "cuSPARSE_GFlops", "DASP_GFlops", "amgT_GFlops", "AtSpMV_GFlops"]:
    df3[c] = df3[c].apply(clean_num)
df3["Matrix"] = df3["Matrix"].str.strip().str.replace('"', '').str.replace(".mtx", "", regex=False)

out3 = os.path.join(OUT, "AtSpMV_H100_Performance.xlsx")
df3.to_excel(out3, index=False, sheet_name="H100 Performance")
print(f"[3] {out3}  —  {len(df3)} matrices")

print("\nDone — 3 Excel files created.")
