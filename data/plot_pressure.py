import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re

plt.figure(figsize=(10, 6))
N_pool = 100
final_pressure = []
d_values = []
for dir_name in os.listdir('.'):
    if os.path.isdir(dir_name):
        # data\N=131072_L=16_D=0.1_T=1_MASS=200_N_TEST=1_T_STEP=0.2_N_STEP=500_bounded=true
        try:
            d = re.search(r'N=131072_L=16_D=([\d.]+)_T=1_MASS=200_N_TEST=1_T_STEP=([\d.]+)_N_STEP=500_bounded=true', dir_name).group(1)
            d_values.append(float(d))
            pressure = pd.read_csv(f'{dir_name}/pressure.csv')
            pooled_pressure = pressure['pressure'].rolling(window=N_pool).mean()
            # plt.plot(pressure['time'], pooled_pressure, label=f'D={d}')
            final_pressure.append(pooled_pressure.iloc[-1])
        except:
            pass
plt.scatter(d_values, final_pressure, label='Final Pressure')
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.legend()
plt.savefig('pressures_bounded=true.png')
