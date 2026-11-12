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
data = pd.read_csv(CSV_DIR / 'fullres.csv')
# fig, ax = plt.subplots(figsize=(16, 8))
x = data['avg_nnz'] * (data['nrow']  - data['zero_len'])
width = 0.25
cusp = data['gflopscusp']
amg = data['gflopsamgt']
dasp = data['gflopsdasp']
dySPMV = data['gflopsdy']
import matplotlib.font_manager as fm
fm.fontManager.addfont('/staff/chengjiajun/workspace/tcspmv/times.ttf')
log10 = np.log10(x)
fig = plt.figure(figsize=(16, 15))
# f, (ax, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(16, 16))
gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
ax = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])

# fig, ax = plt.subplots(figsize=(16, 8))
ax.scatter(log10, cusp, c = 'cyan', s = 6, label = 'cusparse')
ax.scatter(log10, amg, c = 'magenta', s = 6, label = 'amgT')
ax.scatter(log10, dasp, c = 'orange', s = 6, label = 'dasp')
ax.scatter(log10, dySPMV, c = 'red', s = 6, label = 'dySPMV')
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}
ax.legend(fontsize = 20, prop= font2,markerscale=4.0)
ax.margins(x=0)
ax.margins(y=0)
ax.grid(True)
ax.set_ylabel("Performance(GFlops)", fontdict=font2)
ax.tick_params(
    axis='x',          # 针对 X 轴
    which='both',      # 同时影响主副刻度
    bottom=True,       # 保留底部刻度线
    labelbottom=False  # 隐藏底部刻度标签
)

ax1.scatter(log10, data['speedupcusp'], s = 6)
ax1.axhline(1)
ax1.grid(True)
ax1.set_ylabel("speedup over cusparse", fontdict=font2)
ax1.tick_params(
    axis='x',          # 针对 X 轴
    which='both',      # 同时影响主副刻度
    bottom=True,       # 保留底部刻度线
    labelbottom=False  # 隐藏底部刻度标签
)
ax1.margins(x=0)

# ax.margins(x=0)
ax2.scatter(log10, data['speedupdasp'], s = 6)
ax2.grid(True)
ax2.axhline(1)
ax2.set_ylabel("speedup over dasp", fontdict=font2)
ax2.tick_params(
    axis='x',          # 针对 X 轴
    which='both',      # 同时影响主副刻度
    bottom=True,       # 保留底部刻度线
    labelbottom=False  # 隐藏底部刻度标签
)
ax2.margins(x=0)


ax3.scatter(log10, data['speedupamg'], s = 6)
ax3.grid(True)
ax3.axhline(1)
ax3.set_ylabel("speedup over amgT", fontdict=font2)
ax3.margins(x=0)

# ax3.tick_params(
#     axis='x',          # 针对 X 轴
#     which='both',      # 同时影响主副刻度
#     bottom=True,       # 保留底部刻度线
#     labelbottom=False  # 隐藏底部刻度标签
# )
# ax.set_xticks([])
# ax.grid(True, which='major', axis='y', linestyle='-', alpha=0.3)
# ax.grid(True, which='minor', axis='x', linestyle='-', alpha=0.2)
# ax.set_xlabel('')  # 清空X轴标签
# plt.show()

plt.savefig(FIGURE_DIR / 'out.png')