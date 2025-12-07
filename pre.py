import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime
data = pd.read_csv("./gpu_pre_full_clean.csv")
# data = data.replace([np.inf, -np.inf], np.nan).dropna()

df_data = data[data['atPreTime'] != 0.0]
# df_data.to_csv("./gpu_pre_full_clean.csv", index=True)
print(df_data['cuspPreTime/atPreTime'].mean())
print(df_data['dasp_pre_time/atPreTime'].mean())
print(df_data['amgTPretime/atPreTime'].mean())

print(df_data['cuspPreTime/atPreTime'].max())
print(df_data['dasp_pre_time/atPreTime'].max())
print(df_data['amgTPretime/atPreTime'].max())

# print((df_data['cuspPreTime']/df_data['cuspTime']).max())
# print((df_data['amgTPretime']/df_data['cuspTime']).max())
# print((df_data['dasp_pre_time']/df_data['cuspTime']).max())
# print((df_data['atPreTime']/df_data['cuspTime']).mean())

# print((df_data['atPreTime']/df_data['atTime']).mean())
# print((df_data['dasp_pre_time']/df_data['dasp_time']).mean())
# print((df_data['amgTPretime']/df_data['amgTTime']).mean())