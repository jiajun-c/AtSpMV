import pandas as pd
df = pd.read_csv("density_select.csv")
data = df
data = data[data['csr.nnz'] > 1200000]
# data = data[data['csr.nrows'] < 32767]
data = data[data['csr.ncols'] < 32767]
# data = data[data['csr.stand'] > 500]
# data = data[data['csr.stand'] < 1000]

print(data[0:])