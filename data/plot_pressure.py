import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re

plt.figure(figsize=(16, 10))
N_pool = 500
final_pressure = []
d_values = []
pressures = []
for dir_name in os.listdir('.'):
    if os.path.isdir(dir_name):
        # data\N=131072_L=16_D=0.1_T=1_MASS=200_N_TEST=1_T_STEP=0.2_N_STEP=500_bounded=true
        try:
            d = re.search(r'N=131072_L=16_D=([\d.]+)_T=1_MASS=200_N_TEST=1_T_STEP=([\d.]+)_N_STEP=5000_bounded=true', dir_name).group(1)
            d_values.append(float(d))
            pressure = pd.read_csv(f'{dir_name}/pressure.csv')
            pooled_pressure = pressure['pressure'].rolling(window=N_pool).mean()
            pressures.append((d,pressure['time'],pooled_pressure))
            final_pressure.append(pressure['pressure'][:-N_pool].mean())
        except:
            pass
plt.xlabel('Time')
plt.ylabel('Pressure')
pressures.sort(key=lambda x: x[0], reverse=True)
for d, t, pooled_pressure in pressures:
    plt.plot(t, pooled_pressure, label=f'D={d}', linewidth=1)
plt.legend()
plt.savefig('pressure_N_STEP=5000_bounded=true.png')
plt.clf()
ds = np.linspace(min(d_values), max(d_values), 100)
plt.plot(ds, 131072 * 1 / (16 ** 3 - 131072 * 2/3*np.pi*ds**3), 'r', label='theoretical:$p=\\frac{N k T}{V-\\frac{2}{3}\\pi d^3 N}$')
plt.plot(ds, 131072 * 1 / (16 ** 3 - 131072 * 1/12*np.pi*ds**3), 'r--', label='$p=\\frac{N k T}{V-\\pi d^3 N / 12}$')
plt.scatter(d_values, final_pressure, label='Final Pressure')
plt.ylim(min(final_pressure)-0.5, max(final_pressure)+0.5)
plt.xlabel('d')
plt.ylabel('Pressure')
plt.legend()
plt.savefig('pressures_N_STEP=5000_bounded=true.png')
