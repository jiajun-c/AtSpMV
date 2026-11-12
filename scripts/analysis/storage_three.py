from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = ROOT / "results/csv"
FIGURE_DIR = ROOT / "results/figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

data = pd.read_csv(CSV_DIR / 'storage_three.csv')
fm.fontManager.addfont('/staff/chengjiajun/workspace/tcspmv/times.ttf')

data = data[data['at'] != 0]
print(len(data))
ratio = data['at']/data['cusp']
print(ratio.mean())
ratio = data['at']/data['amg']
print(ratio.mean())
# data = data[data['nnz'] > 100000]
print(len(data))
at = np.log10(data['at'])
amg = np.log10(data['amg'])
cusp = np.log10(data['cusp'])
ratio = data['cusp']/data['at']
print(ratio.mean())
x = data['nnz']
log10 = np.log10(x)
fig,ax = plt.subplots(figsize=(20, 10))
fontY = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 40,
}
ax.scatter(log10, cusp, c = 'red', s = 14, label = 'cuSPARSE')
ax.scatter(log10, amg, c = 'magenta', s = 14, label = 'amgT')
ax.scatter(log10, at, c = 'orange', s = 14, label = 'AtSpMV')
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 40,
}
# ax.tick_params(axis='both', which='major', labelsize=20) 
# ax.set_ylim(0, 100)
ax.legend(prop= fontY,markerscale=8.0, loc = 'upper left', frameon=False)
ax.set_xlim(2, log10.max())
for label in ax.get_yticklabels():
    label.set_fontproperties(fontY)
for label in ax.get_xticklabels():
    label.set_fontproperties(fontY)
# ax.margins(x=0)
# ax.margins(y=0)
ax.grid(True)
ax.set_ylabel("Memory Size(log10 scale)B", fontdict=font2)
ax.set_xlabel("log10(nnz)",fontdict=font2)

plt.savefig(FIGURE_DIR / 'storage.pdf', bbox_inches='tight')
plt.savefig(FIGURE_DIR / 'storage.svg', bbox_inches='tight')