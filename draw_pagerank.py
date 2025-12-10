import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
fm.fontManager.addfont('/staff/chengjiajun/workspace/tcspmv/times.ttf')
fm.fontManager.addfont('/staff/chengjiajun/workspace/tcspmv/timeBold.ttf')
matrix = ["cit-HepPh", "cit-HepTh","ca-HepPh","ca-HepTh","web-BerkStan","web-Google","wiki-Vote","wiki-RfA","email-EuAll","email-Enron"]
atPre = [8689,4887,4869,5049,6332,8848,5451,5127,12438,9716]
atIter = [1090, 1578, 2000, 2060, 14408, 11901, 1134, 1785, 2526, 2381]

cuspPre = [8052, 13116, 12064, 7944, 12946, 11092, 15777, 9843, 14796, 9404]
cuspIter = [7135, 10354, 11666, 8076, 13503, 16959, 11804, 7953, 10352, 8713]
fullAtPre = np.array([a + b for a, b in zip(atPre, atIter)])/1000
fullCuspPre = np.array([a + b for a, b in zip(cuspPre, cuspIter)])/1000
speedup = fullCuspPre/fullAtPre
print(speedup)
fig, ax = plt.subplots(figsize=(20, 8))
fontY = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 25,
}
fontLabel = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 30,
}
font3 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 25,
}
x = np.arange(len(matrix))
width = 0.3

setupCusp = np.array(cuspPre) / 1000
iterCusp = np.array(cuspIter) / 1000
setupAt = np.array(atPre) / 1000
iterAt = np.array(atIter) / 1000
fig, ax = plt.subplots(figsize=(20, 8))

# CUSP (noBalance)
p1 = ax.bar(x - 0.5*width, setupCusp, width, 
            label='cuSPARSE:Setup', color="#0FBEC4", edgecolor='black')
p2 = ax.bar(x - 0.5*width, iterCusp, width, 
            bottom=setupCusp, label='cuSPARSE:Iter', color="#0FBEC4", edgecolor='black',hatch='xx')

# AtSpMV (Balance)
p3 = ax.bar(x + 0.5*width, setupAt, width, 
            label='AtSpMV:Setup', color="#0E8286", edgecolor='black')
p4 = ax.bar(x + 0.5*width, iterAt, width, 
            bottom=setupAt, label='AtSpMV:Iter', color="#0E8286", edgecolor='black',hatch='xx')
# ax.bar(x - 0.5*width, fullCuspPre, width=width, label='AtSpMV(noBalance)',color="#0FBEC4",zorder=10, edgecolor='black')
# ax.bar(x + 0.5*width, fullAtPre, width=width, label='AtSpMV(Balance)',color="#0E8286",zorder=10, edgecolor='black')
ax.set_ylabel('Execution Time(ms)',fontdict=fontLabel)
for label in ax.get_yticklabels():
    label.set_fontproperties(fontY)
ax.set_xticks(x)
ax.grid(True, alpha=0.6, axis='y',zorder=0,ls='--')
ax.legend(prop= font3,markerscale=3.0,loc='upper left',ncol=2,framealpha=0)
barFontSize = 25
ax.annotate('1.55x', fontsize=barFontSize,fontweight='bold',
                fontfamily='Times New Roman',xy=(0.125, fullAtPre[0]), xytext=(0.04,fullAtPre[0]+3),
                    color='red',  rotation=90,                  # 文本的颜色
                arrowprops=dict(arrowstyle='<-',color='red', linewidth=2))
ax.annotate('3.63x', fontsize=barFontSize,fontweight='bold',
                fontfamily='Times New Roman',xy=(1.125, fullAtPre[1]), xytext=(1.04,fullAtPre[1]+3),
                    color='red',  rotation=90,                  # 文本的颜色
                arrowprops=dict(arrowstyle='<-',color='red', linewidth=2))
ax.annotate('3.45x', fontsize=barFontSize,fontweight='bold',
                fontfamily='Times New Roman',xy=(2.125, fullAtPre[2]), xytext=(2.04,fullAtPre[2]+3),
                    color='red',  rotation=90,                  # 文本的颜色
                arrowprops=dict(arrowstyle='<-',color='red', linewidth=2))
ax.annotate('2.25x', fontsize=barFontSize,fontweight='bold',
                fontfamily='Times New Roman',xy=(3.125, fullAtPre[3]), xytext=(3.04,fullAtPre[3]+3),
                    color='red',  rotation=90,                  # 文本的颜色
                arrowprops=dict(arrowstyle='<-',color='red', linewidth=2))
ax.annotate('1.27x', fontsize=barFontSize,fontweight='bold',
                fontfamily='Times New Roman',xy=(4.125, fullAtPre[4]), xytext=(4.04,fullAtPre[4]+3),
                    color='red',  rotation=90,                  # 文本的颜色
                arrowprops=dict(arrowstyle='<-',color='red', linewidth=2))
ax.annotate('1.35x', fontsize=barFontSize,fontweight='bold',
                fontfamily='Times New Roman',xy=(5.125, fullAtPre[5]), xytext=(5.04,fullAtPre[5]+3),
                    color='red',  rotation=90,                  # 文本的颜色
                arrowprops=dict(arrowstyle='<-',color='red', linewidth=2))
ax.annotate('4.18x', fontsize=barFontSize,fontweight='bold',
                fontfamily='Times New Roman',xy=(6.125, fullAtPre[6]), xytext=(6.04,fullAtPre[6]+3),
                    color='red',  rotation=90,                  # 文本的颜色
                arrowprops=dict(arrowstyle='<-',color='red', linewidth=2))
ax.annotate('2.57x', fontsize=barFontSize,fontweight='bold',
                fontfamily='Times New Roman',xy=(7.125, fullAtPre[7]), xytext=(7.04,fullAtPre[7]+3),
                    color='red',  rotation=90,                  # 文本的颜色
                arrowprops=dict(arrowstyle='<-',color='red', linewidth=2))
ax.annotate('1.68x', fontsize=barFontSize,fontweight='bold',
                fontfamily='Times New Roman',xy=(8.125, fullAtPre[8]), xytext=(8.04,fullAtPre[8]+3),
                    color='red',  rotation=90,                  # 文本的颜色
                arrowprops=dict(arrowstyle='<-',color='red', linewidth=2))
ax.annotate('1.46x', fontsize=barFontSize,fontweight='bold',
                fontfamily='Times New Roman',xy=(9.125, fullAtPre[9]), xytext=(9.04,fullAtPre[9]+3),
                    color='red',  rotation=90,                  # 文本的颜色
                arrowprops=dict(arrowstyle='<-',color='red', linewidth=2))
ax.set_xticklabels(matrix, fontdict=fontY, weight='bold',rotation=15)
plt.tight_layout()
plt.savefig('pagerank.pdf')
plt.savefig('pagerank.svg')