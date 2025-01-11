from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

speed = pd.read_csv('speed.csv')

# plt.hist(speed['speed'], bins=1000)
# plt.xlabel('Speed')
# plt.title('Speed Distribution')
# plt.savefig('speed_hist.png')

def exp_func(x, a, temp):
    return a * np.exp(-x/temp)

import re

def read_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        content = file.read()
        config['N'] = int(re.search(r'pub const N: usize = (\d+);', content).group(1))
        config['L'] = int(re.search(r'pub const L: usize = (\d+);', content).group(1))
        config['D'] = float(re.search(r'pub const D: f64 = ([\d.]+);', content).group(1))
        config['T'] = float(re.search(r'pub const T: f64 = ([\d.]+);', content).group(1))
        config['MASS'] = float(re.search(r'pub const MASS: f64 = ([\d.]+);', content).group(1))
        config['N_TEST'] = int(re.search(r'pub const N_TEST: usize = (\d+);', content).group(1))
        config['E0'] = float(re.search(r'pub const E0: f64 = ([\d.]+) \* T;', content).group(1)) * config['T']
        config['T_STEP'] = float(re.search(r'pub const T_STEP: f64 = ([\d.]+);', content).group(1))
    return config

config = read_config('../../src/config.rs')
T = config['T']
m = config['MASS']

particle_energies = speed['speed']**2 * 0.5 * m

hist, bins = np.histogram(particle_energies, bins=200)
bin_widths = np.diff(bins)
bin_centers = (bins[:-1] + bins[1:]) / 2
norm_factor = bin_widths * np.sqrt(bin_centers)
normalized_hist = hist / norm_factor
popt, pcov = curve_fit(exp_func, bin_centers, normalized_hist, p0=[1e5, T])
fit_values = exp_func(bin_centers, *popt)
plt.clf()
plt.bar(bin_centers, normalized_hist, width=bin_widths, color='b', alpha=0.5, label='Data')
plt.plot(bin_centers, fit_values, 'r-', label='Fit: a=%.5f, T=%.5f' % tuple(popt))
plt.yscale('log')
plt.xlabel('E')
plt.title(R"$\mathrm{d}N/\sqrt{E}\mathrm{d}E$")
plt.legend()
plt.savefig('energy_hist.png')

pressure = pd.read_csv('pressure.csv')
plt.clf()
plt.plot(pressure['time'], pressure['pressure'], label='Pressure')

# plot pooled pressure
N_pool = 50
pooled_pressure = pressure['pressure'].rolling(window=N_pool).mean()
plt.plot(pressure['time'], pooled_pressure, label='Pooled Pressure')

plt.xlabel('Time')
plt.ylabel('Pressure')
plt.legend()
plt.title('Pressure')
plt.savefig('pressure.png')

