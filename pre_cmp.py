import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
fm.fontManager.addfont('/staff/chengjiajun/workspace/tcspmv/times.ttf')
fm.fontManager.addfont('/staff/chengjiajun/workspace/tcspmv/timeBold.ttf')
matrix = ["IG5-17","mip1","SiO","case9","gen4","pattern1","ship_001","trans4"]
cuspPre = [12702,18702,17045,12005,12420,11770,10511,7870]
atPreBal = [5514,5845,10452,7377,5436,6258,8835,5971]
atPreNoBal = [7120,36675,10778,11116,6424,10685,10099,10335]
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
width = 0.25
p2 = ax.bar(x - width, atPreNoBal, width, 
             label='AtSpMV(no balance preprocess)', color="#BBCBCC", edgecolor='black')
ax.set_ylabel('Preprocessing Time(ms)',fontdict=fontLabel)
p1 = ax.bar(x  , cuspPre, width, 
            label='cuSPARSE-CSR', color="#F7ED92", edgecolor='black')
p2 = ax.bar(x + width, atPreBal, width, 
             label='AtSpMV(balance preprocess)', color="#648788", edgecolor='black')

for label in ax.get_yticklabels():
    label.set_fontproperties(fontY)
ax.set_xticks(x)
ax.grid(True, alpha=0.6, axis='y',zorder=0,ls='--')
ax.legend(prop= font3,markerscale=3.0,loc='upper right',ncol=1,framealpha=0)
ax.set_xticklabels(matrix, fontdict=fontY, weight='bold',rotation=0)
plt.tight_layout()
plt.savefig('pre_cmp.pdf')
plt.savefig('pre_cmp.svg')