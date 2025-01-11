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
        d = re.search(r'D=([\d.]+)', dir_name).group(1)
        d_values.append(float(d))
        pressure = pd.read_csv(f'{dir_name}/pressure.csv')
        pooled_pressure = pressure['pressure'].rolling(window=N_pool).mean()
        # plt.plot(pressure['time'], pooled_pressure, label=f'D={d}')
        final_pressure.append(pooled_pressure.iloc[-1])
plt.scatter(d_values, final_pressure, label='Final Pressure')
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.legend()
plt.savefig('pressures.png')
