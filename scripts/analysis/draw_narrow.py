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
matrix = ['30', '25', '20', '15','10','5']
x = np.arange(len(matrix))
print(x)
width = 0.2
amg = [159, 142, 119, 98, 87, 46]
dasp = [136, 119, 109, 95, 70, 43]
atspmv = [152, 149, 145, 163, 154, 82]

import matplotlib.font_manager as fm
fm.fontManager.addfont('/staff/chengjiajun/workspace/tcspmv/times.ttf')


fig, ax = plt.subplots(figsize=(20, 8))
# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)

# 取消边框
fontY = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}
bars2 = ax.bar(x- width, amg,  width , label='amgT', color="#42d0e9", alpha=0.8,zorder=10, edgecolor='black')
bars3 = ax.bar(x, dasp, width  , label='dasp', color="#CF9553", alpha=0.8,zorder=10, edgecolor='black')
bars3 = ax.bar(x + width, atspmv, width, label='AtSpmv', color="#E9FF98", alpha=0.8,zorder=10, edgecolor='black')
plt.grid(axis='y', linestyle = '--') # 设置 y 就在轴方向显示网格线
for label in ax.get_yticklabels():
    label.set_fontproperties(fontY)


font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 30,
}

fontYLabel = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 30,
}
ax.legend(fontsize = 20, prop= font2,markerscale=4.0)
# ax.grid(True)
ax.set_ylabel("Performance(GFlops)", fontdict=fontYLabel)
# ax.set_yticklabels([0, 20, 40, 80, 100, 120, 140, 160],fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(matrix, ha='center', fontdict=fontY)
ax.set_xlabel("nnz pre row", fontdict=fontYLabel)

plt.savefig(FIGURE_DIR / 'narrow.svg',bbox_inches='tight')
plt.savefig(FIGURE_DIR / 'narrow.pdf',bbox_inches='tight')