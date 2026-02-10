matrix = ["333SP", "Flan_1565", "opt1","JP","crankeg_2","GaAsH6","Zd_Jac6","mip1","torso1","karted"]
stand = [0.0, 11.43, 42.49, 62.47, 95.875702,126.971855,175.482330,350.744,419.587952,823.722900]
bal = [111.36,220.274004,167.883043, 192.091734,197.886126,132.61996,103.756545,184.8717,126.170370,82.341814]
noBal = [109.98,202.598,65.445932, 121.54,40.833,20.129815,17.469214,7.122682,12.307081,1.986924]
import numpy as np
print(np.array(bal)/np.array(noBal))
speedup = [ 1.01254774 , 1.08724669,  2.56521739,  1.5804816,   4.8462304,   6.58823541
  ,5.93939401, 25.9553494,10.25185176 ,41.44185384]
x = np.arange(len(matrix))
fontY = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 25,
}
fontLabel = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 30,
}
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
fm.fontManager.addfont('/staff/chengjiajun/workspace/tcspmv/times.ttf')
fm.fontManager.addfont('/staff/chengjiajun/workspace/tcspmv/timeBold.ttf')
width = 0.3
font3 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 25,
}
fig,ax = plt.subplots(figsize=(18, 7))
ax.bar(x - 0.5*width, noBal, width=width, label='AtSpMV(noBalance)',color="#AEEEEE",zorder=10, edgecolor='black')
ax.bar(x + 0.5*width, bal, width=width, label='AtSpMV(Balance)',color="#668B8B",zorder=10, edgecolor='black')
for label in ax.get_yticklabels():
    label.set_fontproperties(fontY)
ax.set_ylabel('Performance(GFlops)',fontdict=fontLabel)
ax.grid(True, alpha=0.6, axis='y',zorder=0,ls='--')
# ax.margins(y=0.3)
ax.set_ylim(ymin=0, ymax=300)
ax.legend(prop= font3,markerscale=3.0,loc='upper left',ncol=6,framealpha=0)
barFontSize = 20
ax.annotate('8%', fontsize=barFontSize,fontweight='bold',
                fontfamily='Times New Roman',xy=(1.125, bal[1]), xytext=(0.99 ,bal[1]+25),
                    color='red',                   # 文本的颜色
                arrowprops=dict(arrowstyle='<-',color='red', linewidth=2))
ax.annotate('156%', fontsize=barFontSize,fontweight='bold',
                fontfamily='Times New Roman',xy=(2.125, bal[2]), xytext=(1.90 ,bal[2]+30),
                    color='red',                   # 文本的颜色
                arrowprops=dict(arrowstyle='<-',color='red', linewidth=2))
ax.annotate('58%', fontsize=barFontSize,fontweight='bold',
                fontfamily='Times New Roman',xy=(3.125, bal[3]), xytext=(2.95 ,bal[3]+30),
                    color='red',                   # 文本的颜色
                arrowprops=dict(arrowstyle='<-',color='red', linewidth=2))# ax.tick_params(
ax.annotate('384%', fontsize=barFontSize,fontweight='bold',
                fontfamily='Times New Roman',xy=(4.125, bal[4]), xytext=(3.885 ,bal[4]+30),
                    color='red',                   # 文本的颜色
                arrowprops=dict(arrowstyle='<-',color='red', linewidth=2))
ax.annotate('558%', fontsize=barFontSize,fontweight='bold',
                fontfamily='Times New Roman',xy=(5.125, bal[5]), xytext=(4.885 ,bal[5]+30),
                    color='red',                   # 文本的颜色
                arrowprops=dict(arrowstyle='<-',color='red', linewidth=2))
ax.annotate('493%', fontsize=barFontSize,fontweight='bold',
                fontfamily='Times New Roman',xy=(6.125, bal[6]), xytext=(5.89 ,bal[6]+30),
                    color='red',                   # 文本的颜色
                arrowprops=dict(arrowstyle='<-',color='red', linewidth=2))
ax.annotate('2495%', fontsize=barFontSize,fontweight='bold',
                fontfamily='Times New Roman',xy=(7.125, bal[7]), xytext=(6.855 ,bal[7]+30),
                    color='red',                   # 文本的颜色
                arrowprops=dict(arrowstyle='<-',color='red', linewidth=2))
ax.annotate('925%', fontsize=barFontSize,fontweight='bold',
                fontfamily='Times New Roman',xy=(8.125, bal[8]), xytext=(7.885 ,bal[8]+30),
                    color='red',                   # 文本的颜色
                arrowprops=dict(arrowstyle='<-',color='red', linewidth=2),zorder=20)
ax.annotate('4044%', fontsize=barFontSize,fontweight='bold',
                fontfamily='Times New Roman',xy=(9.125, bal[9]), xytext=(8.848 ,bal[9]+30),
                    color='red',                   # 文本的颜色
                arrowprops=dict(arrowstyle='<-',color='red', linewidth=2))
ax.set_xticks(x)
def add_labels(ax, x, values, offset=5, threshold=40):
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
add_labels(ax, x - 0.5*width, noBal)
add_labels(ax, x + 0.5*width, bal)
ax.set_xticklabels(matrix,fontdict=fontY,weight='bold')
ax2 = ax.twinx()
ax2.margins(y=0.3)
ax2.set_ylim(ymin=0, ymax=900)

ax2.plot(x, stand, marker='v', linewidth=2, markersize=15,
         color="#5E005E", zorder=20, label='Standard')  # 折线图
ax2.set_ylabel('Standard Deviation of Row Len', fontdict=fontLabel)
for label in ax2.get_yticklabels():
    label.set_fontproperties(fontY)
plt.tight_layout()

plt.savefig('bar.svg')
plt.savefig('bar.pdf')