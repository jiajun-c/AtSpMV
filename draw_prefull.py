import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.gridspec as gridspec
# %matplotlib inline
data = pd.read_csv("./gpu_pre_full_clean.csv")
# fig, ax = plt.subplots(figsize=(16, 8))
x = data['csr.nnz']
width = 0.25
cusp = data['cuspPreTime']/1000
amg = data['amgTPretime']/1000
dasp = data['dasp_pre_time']/1000
dySPMV = data['atPreTime']/1000
import matplotlib.font_manager as fm
fm.fontManager.addfont('/staff/chengjiajun/workspace/tcspmv/times.ttf')
log10 = np.log10(x)
fig,ax = plt.subplots(figsize=(20, 15))
# f, (ax, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(16, 16))
# gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
# ax = plt.subplot(gs[0])
# ax1 = plt.subplot(gs[1])
# ax2 = plt.subplot(gs[2])
# ax3 = plt.subplot(gs[3])
fontY = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 50,
}
# fig, ax = plt.subplots(figsize=(16, 8))
ax.scatter(log10, cusp, c = 'cyan', s = 11, label = 'cuSPARSE')
ax.scatter(log10, amg, c = 'magenta', s = 11, label = 'amgT')
ax.scatter(log10, dasp, c = 'orange', s = 11, label = 'DASP')
ax.scatter(log10, dySPMV, c = 'red', s = 11, label = 'AtSpMV')
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 60,
}
# ax.tick_params(axis='both', which='major', labelsize=20) 
ax.set_ylim(0, 100)
ax.legend(prop= fontY,markerscale=8.0, loc = 'upper left')
ax.set_xlim(2, log10.max())
for label in ax.get_yticklabels():
    label.set_fontproperties(fontY)
for label in ax.get_xticklabels():
    label.set_fontproperties(fontY)
ax.margins(x=0)
ax.margins(y=0)
ax.grid(True)
ax.set_ylabel("Preprocessing Overhead(ms)", fontdict=font2)
ax.set_xlabel("log10(nnz)",fontdict=font2)
# ax.tick_params(
#     axis='x',          # 针对 X 轴
#     which='both',      # 同时影响主副刻度
#     bottom=True,       # 保留底部刻度线
#     labelbottom=False  # 隐藏底部刻度标签
# )



plt.savefig("./pre_nnz.pdf", bbox_inches='tight')
plt.savefig("./pre_nnz.svg", bbox_inches='tight')