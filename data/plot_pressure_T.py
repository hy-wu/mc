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
MASS = 200.0
final_Ts = []
l_j = []
ns = []
for dir_name in os.listdir('.'):
    if os.path.isdir(dir_name):
        # data\N=131072_L=16_D=0.1_T=1_MASS=200_N_TEST=1_T_STEP=0.2_N_STEP=500_bounded=true
        pattern1 = r'N=16384_L=8_D=0.2_T=([\d.]*)_MASS=200_N_TEST=1_T_STEP=([\d.]*)_EPS=1_N_STEP=10000_bounded=true'
        pattern1_new = r'N=16384_L=8_D=0.2_T=([\d.]*)_MASS=200_N_TEST=1_T_STEP=([\d.]*)_EPS=1_N_STEP=20000_bounded=true'
        pattern0 = r'N=16384_L=8_D=0.2_T=([\d.]*)_MASS=200_N_TEST=1_T_STEP=([\d.]*)_EPS=0_N_STEP=10000_bounded=true'
        pattern_old = r'N=16384_L=8_D=0.2_T=([\d.]*)_MASS=200_N_TEST=1_T_STEP=([\d.]*)_N_STEP=10000_bounded=true'
        # if re.search(pattern1, dir_name) or re.search(pattern0, dir_name) or re.search(pattern_old, dir_name):
        if re.search(pattern1, dir_name):
            T, dt = re.search(pattern1, dir_name).groups()
            l_j.append(True)
            ns.append(10000)
        elif re.search(pattern_old, dir_name):
            T, dt = re.search(pattern_old, dir_name).groups()
            l_j.append(False)
            ns.append(10000)
        elif re.search(pattern0, dir_name):
            T, dt = re.search(pattern0, dir_name).groups()
            l_j.append(False)
            ns.append(10000)
        elif re.search(pattern1_new, dir_name):
            T, dt = re.search(pattern1_new, dir_name).groups()
            l_j.append(True)
            ns.append(20000)
            print(f'Found new pattern: {dir_name}')
        else:
            continue
        T_values.append(float(T))
        pressure = pd.read_csv(f'{dir_name}/pressure.csv')
        pooled_pressure = pressure['pressure'].rolling(window=N_pool).mean()
        pressures.append((T,pressure['time'],pooled_pressure))
        final_pressure.append(pressure['pressure'][:-N_pool].mean())
        v = pd.read_csv(f'{dir_name}/speed.csv')
        final_T = v['speed'].values.mean()**2 / 3 * MASS
        final_Ts.append(final_T)
T_values, final_pressure, final_Ts, l_j, ns = zip(*sorted(zip(T_values, final_pressure, final_Ts, l_j, ns), key=lambda x: x[0], reverse=True))
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.xscale('log')
plt.yscale('log')
pressures.sort(key=lambda x: x[0], reverse=True)
for i, (T, t, pooled_pressure) in enumerate(pressures):
    if ns[i] == 20000:
        plt.plot(t, pooled_pressure, '--', label=f'T={T} L_J={l_j[i]} N_STEP={ns[i]}', linewidth=1)
    else:
        plt.plot(t, pooled_pressure, label=f'T={T} L_J={l_j[i]} N_STEP={ns[i]}', linewidth=1)
plt.legend()
plt.savefig('pressure_N=16384_L=8_D=0.2_EPS=1_N_STEP=10000_bounded=true.png')
plt.clf()
final_pressure_LJ = []
T_values_LJ = []
final_Ts_LJ = []
final_pressure_LJ_new = []
T_values_LJ_new = []
final_Ts_LJ_new = []
final_pressure_nonLJ = []
T_values_nonLJ = []
final_Ts_nonLJ = []
for i, (T, p, T_final, lj) in enumerate(zip(T_values, final_pressure, final_Ts, l_j)):
    if p != 0:
        if lj:
            if ns[i] == 20000:
                final_pressure_LJ_new.append(p)
                T_values_LJ_new.append(T)
                final_Ts_LJ_new.append(T_final)
            else:
                final_pressure_LJ.append(p)
                T_values_LJ.append(T)
                final_Ts_LJ.append(T_final)
        else:
            final_pressure_nonLJ.append(p)
            T_values_nonLJ.append(T)
            final_Ts_nonLJ.append(T_final)
Ts = np.linspace(min(T_values), max(T_values), 100)

# 创建主轴用于压强
fig, ax1 = plt.subplots(figsize=(12, 8))

# 绘制压强相关的图
ax1.plot(Ts, 16384 * Ts / (8 ** 3 - 16384 * 2/3*np.pi*0.2**3), 'r', label='theoretical:$p=\\frac{N k T}{V-\\frac{2}{3}\\pi d^3 N}$')
ax1.plot(Ts, 16384 * Ts / (8 ** 3 - 16384 * 1/12*np.pi*0.2**3), 'r--', label='$p=\\frac{N k T}{V-\\pi d^3 N / 12}$')
ax1.scatter(T_values_nonLJ, final_pressure_nonLJ, label='Final Pressure (non-LJ)', color='green')
ax1.scatter(T_values_LJ, final_pressure_LJ, label='Final Pressure (LJ)', color='blue')
print(T_values_LJ_new, final_pressure_LJ_new)
ax1.scatter(T_values_LJ_new, final_pressure_LJ_new, label='Final Pressure (LJ 20000 steps)', color='cyan')

ax1.set_xlabel('T')
ax1.set_ylabel('Pressure')  # , color='blue'
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.tick_params(axis='y')  # , labelcolor='blue'
ax1.set_ylim(np.exp(min(np.log(final_pressure)) - 0.5), np.exp(max(np.log(final_pressure)) + 0.5))

# 创建右侧y轴用于温度
ax2 = ax1.twinx()
ax2.plot(Ts, Ts, label='Initial Temperature', color='orange', linestyle='--')
ax2.scatter(T_values_nonLJ, final_Ts_nonLJ, label='Final Temperature (non-LJ)', marker='+', color='purple', s=50)
ax2.scatter(T_values_LJ, final_Ts_LJ, label='Final Temperature (LJ)', marker='x', color='orange', s=50)
ax2.scatter(T_values_LJ_new, final_Ts_LJ_new, label='Final Temperature (LJ 20000 steps)', marker='x', color='red', s=50)
ax2.set_ylabel('Final Temperature')  # , color='orange'
ax2.set_yscale('log')
ax2.tick_params(axis='y')  # , labelcolor='orange'

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

print(f'Final pressures: {final_pressure}')
print(f'T values: {T_values}')
print(f'Final Ts: {final_Ts}')
plt.savefig('pressures_N=16384_L=8_D=0.2_EPS=1_N_STEP=10000_bounded=true.png')
