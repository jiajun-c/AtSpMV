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
# %matplotlib inline
# fig, ax = plt.subplots(figsize=(16, 8))
matrix = ['af_shell3', 'wiki-Talk', 'mip1', 'human_gene2','144','appu']
x = np.arange(len(matrix))
print(x)
width = 0.2
cusp = [7.75, 8, 7.9,  7.47, 7.8, 7.4]
amg = [25, 164, 128, 64, 33, 8]
atspmv = [4.715, 6.625, 3.75, 4.2, 4.42, 4.2]
dasp = [141.5, 119, 141, 98, 23, 15]
import matplotlib.font_manager as fm
fm.fontManager.addfont('/staff/chengjiajun/workspace/tcspmv/times.ttf')
fig, ax = plt.subplots(figsize=(20, 8))

bars2 = ax.bar(x- 1.5*width, amg,  width , label='amgT', color="#d36200", alpha=0.8)
bars3 = ax.bar(x - 0.5*width, dasp, width  , label='dasp', color="#70706D", alpha=0.8)
bars1 = ax.bar(x + 0.5*width, cusp, width, label='cusparse', color='#1f77b4', alpha=0.8)
bars3 = ax.bar(x + 1.5*width, atspmv, width, label='AtSpmv', color="#9C1BE7", alpha=0.8)
plt.grid(axis='y') # 设置 y 就在轴方向显示网格线



font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 30,
}
fontY = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 30,
}
ax.legend(fontsize = 20, prop= font2,markerscale=4.0)
# ax.grid(True)
ax.set_ylabel("preprocess time(ms)", fontdict=fontY)
ax.set_xticks(x)
ax.set_xticklabels(matrix, ha='center', fontsize=15, fontdict=font2)


plt.savefig(FIGURE_DIR / 'pre.svg')