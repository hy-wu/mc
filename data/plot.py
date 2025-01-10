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

T = 1
m = 200

particle_energies = speed['speed']**2 * 0.5 * m

hist, bins = np.histogram(particle_energies, bins=200)
bin_widths = np.diff(bins)
bin_centers = (bins[:-1] + bins[1:]) / 2
norm_factor = bin_widths * np.sqrt(bin_centers)
normalized_hist = hist / norm_factor
popt, pcov = curve_fit(exp_func, bin_centers, normalized_hist)
fit_values = exp_func(bin_centers, *popt)
plt.clf()
plt.bar(bin_centers, normalized_hist, width=bin_widths, color='b', alpha=0.5, label='Data')
plt.plot(bin_centers, fit_values, 'r-', label='Fit: a=%.5f, T=%.5f' % tuple(popt))
plt.yscale('log')
plt.xlabel('E')
plt.title(R"$\mathrm{d}N/\sqrt{E}\mathrm{d}E$")
plt.legend()
plt.savefig('energy_hist.png')
