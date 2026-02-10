import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.gridspec as gridspec

# %matplotlib inline
# data = pd.read_csv("./fullres.csv")
# fig, ax = plt.subplots(figsize=(16, 8))
width = 0.25
matrix = ['nw14','tp-6','wiki-Talk', 'landmark','specular','boyd1','Raj1','c-big','TF19','a0nsdsil','karted','mip1','connectus','us04','human_gen1','n4c6-b7','Maragal_6','pkustk04','language','333SP','circuit5M','net4-1','karted','lp_osa_60']

import matplotlib.font_manager as fm
fm.fontManager.addfont('/staff/chengjiajun/workspace/tcspmv/times.ttf')
fm.fontManager.addfont('/staff/chengjiajun/workspace/tcspmv/timeBold.ttf')

fig = plt.figure(figsize=(28, 12))
# f, (ax, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(16, 16))
gs = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 2])
ax = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1],sharex=ax)
ax2 = plt.subplot(gs[2],sharex=ax)
# ax3 = plt.subplot(gs[3])
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}
fontLabel = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 28,
}
fontY = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}
fontX = {'family' : 'Times New Roman',
'size'   : 25,
}
density1x32 = [0.998709,0.823849,0.548687,0.50,0.500000,0.341800,0.152233,0.206850,0.525058,0.136528,0.711857,0.891312,0.993099,0.990631,0.985323,0.250000,0.468388,0.851949,0.094095,0.186998,0.283234,0.637593,0.711857,0.889942]
density2x16 = [0.881800,0.914688,0.710684,1.0,1.0000,0.556051,0.296692,0.384361,0.794117,0.268926,0.836965,0.951677,0.926534,0.936549,0.992513,0.499997,0.654524,0.913858,0.184307,0.373996,0.490266,0.755231,0.836965,0.846149]
density4x8 = [0.719828,0.826508,0.804726,1.0,1.000000,0.776303,0.551495,0.593543,0.863375,0.522011,0.810338,0.959288,0.791766,0.913545,0.995965,0.999994,0.810210,0.966207,0.349531,0.746917,0.740735,0.898360,0.810338,0.804841 ]
density8x4 = [0.514724,0.658642,0.807795,1.0,1.00000,0.739450,0.679896,0.765909,0.931958,0.739556,0.682365,0.950277,0.621727,0.767610,0.995929,0.999994,0.906405,0.978695,0.610257,0.753732,0.754077,0.953573,0.682365,0.633756 ]
density16x2 = [0.421349,0.460388,0.720367,1.0,0.999983,0.536315,0.654720,0.859301,0.978999,0.771706,0.498252,0.912453,0.424663,0.562598,0.988566,0.999994,0.848676,0.978288,0.817464,0.950834,0.742147,0.951136,0.498252,0.379720]
density32x1 = [0.238313,0.285855,0.622043,0.999778,0.999983,0.333072,0.518167,0.796479,0.999130,0.747276,0.319508,0.837641,0.255490,0.366513,0.995929,0.999896,0.717032,0.958940,0.779238,0.999963,0.659658,0.934800,0.319508,0.209577 ]

score1x32 = [0.897273,0.739568,0.490971,0.446875,0.446875,0.303049,0.126746,0.178612,0.469577,0.111431,0.638477,0.800427,0.892216,0.889991,0.885205,0.218750,0.418213,0.764920,0.068080,0.159943,0.249394,0.571383,0.638477,0.799192]
score2x16 = [0.792202,0.821852,0.637857,0.898750,0.898750,0.498198,0.262810,0.342673,0.713132,0.237385,0.751775,0.855196,0.832531,0.841560,0.892002,0.447497,0.587162,0.821104,0.159094,0.333254,0.438689,0.678052,0.751775,0.760057]
score4x8 = [0.646543,0.742723,0.723088,0.899062,0.899062,0.697465,0.494646,0.532610,0.775951,0.468014,0.728147,0.862382,0.711405,0.821164,0.895427,0.899057,0.728032,0.868616,0.311896,0.670970,0.665396,0.807480,0.728147,0.723192]
score8x4 = [0.462038,0.591828,0.726242,0.899375,0.899375,0.664660,0.610988,0.688502,0.838092,0.664755,0.613212,0.854592,0.558549,0.690035,0.895708,0.899369,0.815075,0.880187,0.610257,0.677529,0.677840,0.857560,0.613212,0.569394]
score16x2 = [0.378473,0.413670,0.647897,0.899687,0.899672,0.482101,0.588771,0.773007,0.880780,0.694130,0.447800,0.820865,0.381461,0.505783,0.889709,0.899369,0.763440,0.880140,0.817464,0.855422,0.667511,0.855694,0.447800,0.340925]
score32x1 = [0.214482,0.257270,0.559839,0.899800,0.899985,0.299765,0.466350,0.716831,0.899217,0.672549,0.287557,0.753877,0.229941,0.329861,0.895708,0.899906,0.645329,0.863046,0.779238,0.899967,0.593692,0.841320,0.287557,0.188619]

flops1x32 = [90.491000,114.231871,55.485193,63.957333,61.674323,36.156149,17.482738,24.259181,61.996043,16.137909,72.259143,168.338520,83.520370,42.505429,218.315425,26.378182,46.756000,124.078235,12.539526,24.387778,36.720722,66.896630,72.259143,82.827824]
flops2x16 = [86.181905,134.940573,64.792387,88.556308,97.421860, 52.662217,26.854928,36.866315,80.196716,25.359571,98.352722,181.628404,77.760345,39.671733,222.249036,46.632857,53.769400,143.005424,19.618290,43.520599,78.997068,90.434333,90.787128,82.827824]
flops4x8 = [72.392800,106.827954,73.305255,109.641143,129.620610,69.213200,44.912552,57.802741,105.318578,35.503400,90.787128,186.537279,62.640278,37.192250,214.518635,96.720000,53.769400,159.194717,32.435573,83.210734,111.887765,113.568698,90.787128,70.403650]
flops8x4 = [53.230000,90.136086,73.305255,135.439059,198.639377,62.114410,68.550737,83.607536,126.687565,41.768706,68.090346,180.049026,45.101000,37.192250,268.148293,113.540870,63.258118,162.256154,60.816700,97.018629,125.051032,147.983455,69.425451,54.156654]
flops16x2 = [32.318214,33.983561,60.136647,135.439059,191.190400,29.542220,48.239407,95.551469,130.469284,41.768706,43.712321,130.224138,19.110593,28.336952,255.643969,124.354286,51.208952,159.194717,81.088933,116.932979,114.359829,143.631000,43.712321,29.643642]
flops32x1 = [14.834590,9.691238,36.255668,135.439059,198.639377,8.938974,19.884947,60.025923,130.469284,35.503400,31.898180,70.909719,6.316667,14.514049,220.264670,130.572000,37.082345,143.005424,56.573674,126.234466,60.492166,125.216769,29.753765,16.278301]

# final = ['1x32','2x16','8x4','32x1','32x1','4x8','8x4','16x2','32x1','16x2','4x8','1x32','1x32','8x4','32x1']
atflops = [90.491000,135.813497,73.305255,135.439059,198.639377,69.213200,68.550737,95.551469,130.469284,41.768706,98.352722,186.537279,83.520370,42.505429,268.148293,130.572000,63.258118,162.256154,81.088933,126.234466,124.788870,147.983455,90.787128,82.827824]
daspflops = [19.888132,131.856217,66.952133,92.098560,135.356035,50.467958,34.732373,67.855391,106.602951,20.884353,69.425451,136.221303,29.671711,12.397417,191.981658,66.960000,18.541172,94.801348,41.942552,83.523556,108.225984,93.912577,63.226750,54.156654]
cusparseFlops = [67.030370,142.437272,9.721994,74.273032,128.531361,78.143935,70.403459,97.542125,113.525221, 29.586167,84.302333,136.221303,29.671711,29.753800,160.714287,74.612571,41.361077,122.280000,71.549059,112.492486,115.134025,95.754000,84.302333,78.226278]
font3 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 20,
}

x = np.arange(len(matrix))
print(x)
width = 0.25
print(len(x))
ax.bar(x - width, atflops, width, label='AtSpMV',  color="#AEEEEE",zorder=10, edgecolor='black')
ax.bar(x,cusparseFlops , width, label='cuSPARSE', color="#668B8B",zorder=10, edgecolor='black')
ax.bar(x + width, daspflops, width, label='DASP',  color="#F7FFAE",zorder=10, edgecolor='black')
ax.margins(y=0.2)

def add_labels(ax, x, values, offset=5, threshold=40):
    for xi, v in zip(x, values):
        if v < threshold:  
            # 值太小 → 放在柱子上面
            ax.text(
                xi, v + offset, f"{v:.1f}",
                ha='center', va='bottom',
                fontsize=18, fontweight='bold',
                fontfamily='Times New Roman',
rotation='vertical',
                zorder=15
            )
        else:
            # 值足够大 → 放在柱子内部（白字）
            ax.text(
                xi, v - offset, f"{v:.1f}",
                ha='center', va='top',
                fontsize=18, fontweight='bold',
                color='black',
                fontfamily='Times New Roman',
        rotation='vertical',
                zorder=15
            )

# 每一组柱子添加数据
add_labels(ax, x - width, atflops)
add_labels(ax, x, cusparseFlops)
add_labels(ax, x + width, daspflops)



# ax.set_yticks(fontproperties=font2)
for label in ax.get_yticklabels():
    label.set_fontproperties(fontY)
ax.set_ylabel('Performance(GFlops)',fontdict=fontLabel)
# ax.set_title('Performance Comparison of Different Methods')
ax.set_xticks(x)
ax.legend(prop= font3,markerscale=4.0,loc='upper left',ncol=3)
ax.grid(True, alpha=0.6, axis='y',zorder=0,ls='--')
ax.tick_params(
    axis='x',          # 针对 X 轴
    which='both',      # 同时影响主副刻度
    bottom=True,       # 保留底部刻度线
    labelbottom=False  # 隐藏底部刻度标签
)

ax.margins(x=0.01)
ax1.margins(x=0.01)
ax2.margins(x=0.01)

ax1.set_ylabel('Performance(GFlops)',fontdict=fontLabel)
ax1.plot(x, flops1x32, marker='o',label='1x32',color="#61AF7B", markersize=8)
ax1.plot(x, flops2x16, marker='v',label='2x16',color="#C43434", markersize=8)
ax1.plot(x, flops4x8, marker='^',label='4x8',color="#773B54", markersize=8)
ax1.plot(x, flops8x4, marker='p',label='8x4',color="#00B63D", markersize=8)
ax1.plot(x, flops16x2, marker='X',label='16x2',color="#1FAEF5", markersize=8)
ax1.plot(x, flops32x1, marker='d',label='32x1',color="#534F7E", markersize=8)
ax1.tick_params(
    axis='x',          # 针对 X 轴
    which='both',      # 同时影响主副刻度
    bottom=True,       # 保留底部刻度线
    labelbottom=False  # 隐藏底部刻度标签
)
ax1.legend(prop= font3,markerscale=3.0,loc='upper left',ncol=6,framealpha=0)
ax1.margins(y=0.25)
ax1.set_ylim(bottom=0)

ax1.grid(True, alpha=0.6, axis='y',zorder=0,ls='--')
ax1.tick_params(axis='y', labelsize=10)
for label in ax1.get_yticklabels():
    label.set_fontproperties(fontY)
ax2.set_ylabel('Score',fontdict=fontLabel)
ax2.plot(x, score1x32, marker='o',label='1x32',color="#61AF7B", markersize=8)
ax2.plot(x, score2x16, marker='v',label='2x16',color="#C43434", markersize=8)
ax2.plot(x, score4x8, marker='^',label='4x8',color="#773B54", markersize=8)
ax2.plot(x, score8x4, marker='p',label='8x4',color="#00B63D", markersize=8)
ax2.plot(x, score16x2, marker='X',label='16x2',color="#1FAEF5", markersize=8)
ax2.plot(x, score32x1, marker='d',label='32x1',color="#534F7E", markersize=8)
ax2.set_xticklabels(matrix,fontdict=fontX,rotation=45,weight='bold')
ax2.margins(y=0.4)
ax2.set_ylim(bottom=0)
ax2.legend(prop= font3,markerscale=3.0,loc='upper left',ncol=6,framealpha=0)
ax2.grid(True, alpha=0.6, axis='y',zorder=0,ls='--')
for label in ax2.get_yticklabels():
    label.set_fontproperties(fontY)
plt.tight_layout()
plt.savefig('analysis_results.svg')
plt.savefig('analysis_results.pdf')