import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re

plt.figure(figsize=(16, 10))
N_pool = 500
final_pressure = []
T_values = []
pressures = []
for dir_name in os.listdir('.'):
    if os.path.isdir(dir_name):
        # data\N=131072_L=16_D=0.1_T=1_MASS=200_N_TEST=1_T_STEP=0.2_N_STEP=500_bounded=true
        try:
            T = re.search(r'N=16384_L=8_D=0.2_T=([\d.]*)_MASS=200_N_TEST=1_T_STEP=([\d.]*)_N_STEP=10000_bounded=true', dir_name).group(1)
            T_values.append(float(T))
            pressure = pd.read_csv(f'{dir_name}/pressure.csv')
            pooled_pressure = pressure['pressure'].rolling(window=N_pool).mean()
            pressures.append((T,pressure['time'],pooled_pressure))
            final_pressure.append(pressure['pressure'][:-N_pool].mean())
        except:
            pass
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.xscale('log')
plt.yscale('log')
pressures.sort(key=lambda x: x[0], reverse=True)
for T, t, pooled_pressure in pressures:
    plt.plot(t, pooled_pressure, label=f'T={T}', linewidth=1)
plt.legend()
plt.savefig('pressure_N=16384_L=8_D=0.2_N_STEP=10000_bounded=true.png')
plt.clf()
final_pressure = [p for p in final_pressure if p != 0]
T_values = [T for T, p in zip(T_values, final_pressure) if p != 0]
Ts = np.linspace(min(T_values), max(T_values), 100)
# plt.plot(Ts, 131072 * Ts / (16 ** 3 - 131072 * 2/3*np.pi*0.2**3), 'r', label='theoretical:$p=\\frac{N k T}{V-\\frac{2}{3}\\pi d^3 N}$')
plt.plot(Ts, 131072 * Ts / (16 ** 3 - 131072 * 1/12*np.pi*0.2**3), 'r--', label='$p=\\frac{N k T}{V-\\pi d^3 N / 12}$')
plt.scatter(T_values, final_pressure, label='Final Pressure')
print(f'Final pressures: {final_pressure}')
print(f'T values: {T_values}')
plt.ylim(np.exp(min(np.log(final_pressure)) - 0.5), np.exp(max(np.log(final_pressure)) + 0.5))
plt.xlabel('T')
plt.ylabel('Pressure')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('pressures_N=16384_L=8_D=0.2_N_STEP=10000_bounded=true.png')
