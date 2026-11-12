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
fm.fontManager.addfont('/staff/chengjiajun/workspace/tcspmv/times.ttf')
fm.fontManager.addfont('/staff/chengjiajun/workspace/tcspmv/timeBold.ttf')
# matrix = ["IG5-17","SiO","case9","gen4","pattern1","ship_001","trans4","mip1"]
# cuspPre = [12702,17045,12005,12420,11770,10511,7870,18702]
# atPreBal = [5514,10452,7377,5436,6258,8835,5971,5845]
# atPreNoBal = [7120,10778,11116,6424,10685,10099,10335,36675]

matrix = ['landmark','TF18','ex11','opt1','SiO','Raj1','c-73','boyd2']
cuspPre = [7416, 7707, 7944, 7719, 7589, 7798, 7910,8698]
atPreNoBal = [4215.00, 4911, 5358, 4932, 6041, 25732,28815 ,56457]
atPreBal = [4144, 4491, 4357, 4217, 4691 ,6214, 5858,6190]
stand = [1.349165, 8.681529, 16.231871, 42.494785,58.020428, 88.291542,151.4940,194.912064]
cuspPre = np.array(cuspPre)/1000
atPreBal = np.array(atPreBal)/1000
atPreNoBal = np.array(atPreNoBal)/1000
fig, ax = plt.subplots(figsize=(18, 8))
fontY = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 30,
}
fontLabel = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 35,
}
font3 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 25,
}
x = np.arange(len(matrix))
def add_labels(ax, x, values, offset=1, threshold=5):
    for xi, v in zip(x, values):
        if v < threshold:  
            # 值太小 → 放在柱子上面
            ax.text(
                xi, v + offset, f"{v:.1f}",
                ha='center', va='bottom',
                fontsize=25, fontweight='bold',
                fontfamily='Times New Roman',
rotation='vertical',
                zorder=15
            )
        else:
            # 值足够大 → 放在柱子内部（白字）
            ax.text(
                xi, v - offset, f"{v:.1f}",
                ha='center', va='top',
                fontsize=25, fontweight='bold',
                color='black',
                fontfamily='Times New Roman',
        rotation='vertical',
                zorder=15
            )

width = 0.25
add_labels(ax, x - width, atPreNoBal);
add_labels(ax, x + width, cuspPre);
add_labels(ax, x , atPreBal);
# barFontSize = 20
# ax.annotate('8%', fontsize=barFontSize,fontweight='bold',
#                 fontfamily='Times New Roman',xy=(0, atPreBal[0]), xytext=(-0.125,atPreBal[0]+2),
#                     color='red',                   # 文本的颜色
#                 arrowprops=dict(arrowstyle='<-',color='red', linewidth=2))
ax2 = ax.twinx()
# ax2.set_ylabel('Speedup Ratio (noBal/Bal)', fontdict=fontLabel, color='red')
log2stand = np.log2(np.array(stand))
speedup_line = ax2.plot(x, log2stand, marker='v', linewidth=3, markersize=10, 
                        label='Speedup Ratio', zorder=10, markerfacecolor="#3A3470")
p2 = ax.bar(x - width, atPreNoBal, width, 
             label='AtSpMV(no balance preprocess)', color="#BBCBCC", edgecolor='black')
ax.set_ylabel('Preprocessing Time(ms)',fontdict=fontLabel)
p1 = ax.bar(x + width , cuspPre, width, 
            label='cuSPARSE-CSR', color="#F7ED92", edgecolor='black')
p2 = ax.bar(x , atPreBal, width, 
             label='AtSpMV(balance preprocess)', color="#648788", edgecolor='black')
ax2.set_ylabel('log2(standard deviation)',fontdict=fontLabel)

for label in ax.get_yticklabels():
    label.set_fontproperties(fontY)
for label in ax2.get_yticklabels():
    label.set_fontproperties(fontY)
ax.set_xticks(x)
ax.grid(True, alpha=0.7, axis='y',zorder=0,ls='--')
ax.legend(prop= font3,markerscale=3.0,loc='upper left',ncol=1,framealpha=0)
ax.set_xticklabels(matrix, fontdict=fontY, weight='bold',rotation=0)
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'pre_cmp.pdf')
plt.savefig(FIGURE_DIR / 'pre_cmp.svg')