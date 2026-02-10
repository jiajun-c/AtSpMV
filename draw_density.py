import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.gridspec as gridspec
# %matplotlib inline
data = pd.read_csv("./density_select.csv")
data = data[data['index']!=-1]
# fig, ax = plt.subplots(figsize=(16, 8))
x = data['csr.nnz']
width = 0.25
cusp = data['2x2']
amg = data['4x4']
dasp = data['8x8']
dySPMV = data['density']
fontY = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 40,
}
fontD = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 40,
'color'  : 'red'
}
import matplotlib.font_manager as fm
fm.fontManager.addfont('/staff/chengjiajun/workspace/tcspmv/times.ttf')
log10 = np.log10(x)
fig,ax = plt.subplots(figsize=(20, 10))
ax.axhspan(0.8, 1.0, facecolor='none', hatch='////', edgecolor='gray', alpha=0.5)
ax.text(7, 0.9, "High density", fontdict=fontD)
# f, (ax, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(16, 16))
# gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
# ax = plt.subplot(gs[0])
# ax1 = plt.subplot(gs[1])
# ax2 = plt.subplot(gs[2])
# ax3 = plt.subplot(gs[3])


# fig, ax = plt.subplots(figsize=(16, 8))
ax.scatter(log10, cusp, c = 'cyan', s = 6, label = 'BSR(2x2)')
ax.scatter(log10, amg, c = 'magenta', s = 6, label = 'BSR(4x4)')
ax.scatter(log10, dasp, c = 'orange', s = 6, label = 'BSR(8x8)')
ax.scatter(log10, dySPMV, c = 'red', s = 6, label = 'AtBSR')
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 40,
}
ax.tick_params(axis='both', which='major', labelsize=20) 
ax.set_ylim(0, 1)
ax.set_xlim(2, log10.max())
for label in ax.get_yticklabels():
    label.set_fontproperties(fontY)
for label in ax.get_xticklabels():
    label.set_fontproperties(fontY)
ax.legend(prop= font2,markerscale=5.0,frameon=False,loc='upper left')
ax.margins(x=0)
# ax.margins(y=0)
ax.grid(True)
ax.set_ylabel("density", fontdict=font2)
ax.set_xlabel("log10(nnz)",fontdict=font2)
# ax.tick_params(
#     axis='x',          # 针对 X 轴
#     which='both',      # 同时影响主副刻度
#     bottom=True,       # 保留底部刻度线
#     labelbottom=False  # 隐藏底部刻度标签
# )



plt.savefig("./density.svg", bbox_inches='tight')
plt.savefig("./density.pdf", bbox_inches='tight')

print(len(data[data['density']>0.8]))
# for i in range(0, 6):
#     print(len(data[data['index']==i]))
