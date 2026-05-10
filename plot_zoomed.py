import json
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("eos_analysis/eps_fine_results.csv")
eps_arr = df["eps"].values
Z_arr = df["Z"].values
Z_sem_arr = df["Z_sem"].values
Z_baseline = Z_arr[0]
rho = 16384 / 8**3
T = 1.0
sigma = 0.2

from scipy import integrate
def lj_potential(r, sigma, epsilon):
    sr6 = (sigma / r) ** 6
    return 4.0 * epsilon * (sr6**2 - sr6)

def compute_B2_lj(T, sigma, epsilon):
    r_max = math.sqrt(10.0) * sigma
    def integrand(r):
        u = lj_potential(r, sigma, epsilon)
        return (math.exp(-u / T) - 1.0) * r**2
    r_inner = max(0.01 * sigma, 0.3 * sigma)
    result1, _ = integrate.quad(integrand, r_inner, sigma, limit=200)
    result2, _ = integrate.quad(integrand, sigma, r_max, limit=200)
    return -2.0 * math.pi * (result1 + result2)

def compute_B2_hard_sphere(sigma):
    return 2.0 * math.pi * sigma**3 / 3.0

eps_theory = np.linspace(0.001, 2.5, 500)
B2_theory = np.array([compute_B2_lj(T, sigma, e) for e in eps_theory])
B2_hs = compute_B2_hard_sphere(sigma)
Z_virial = 1.0 + B2_theory * rho

mask = eps_arr <= 2.5

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.errorbar(eps_arr[mask], Z_arr[mask], yerr=Z_sem_arr[mask], fmt="ro-", ms=6, capsize=3, lw=1.5, label="Simulation")
ax.plot(eps_theory, Z_virial, "b--", lw=2, alpha=0.7, label=r"Virial: $Z = 1 + B_2(\epsilon)\rho$")
ax.axhline(y=1.0, color="gray", ls="-", lw=0.5, alpha=0.5)
ax.axhline(y=Z_baseline, color="orange", ls="--", lw=1, label=f"Hard sphere Z={Z_baseline:.3f}")
ax.set_xlabel(r"Lennard-Jones $\epsilon$", fontsize=14)
ax.set_ylabel("Compressibility factor Z", fontsize=14)
ax.set_title("Z vs ε (Normal Range ε ≤ 2.5)", fontsize=13)
ax.legend(fontsize=10, loc="best")
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.1, 2.6)
ax.set_ylim(min(-5.0, np.min(Z_virial)), max(2.5, np.max(Z_arr[mask])*1.1))

ax = axes[1]
B2_eff_sim = (Z_arr[mask] - 1.0) / rho
B2_eff_sem = Z_sem_arr[mask] / rho
ax.errorbar(eps_arr[mask], B2_eff_sim, yerr=B2_eff_sem, fmt="ro-", ms=6, capsize=3, lw=1.5, label=r"Simulation $B_2^{eff}$")
ax.plot(eps_theory, B2_theory, "b--", lw=2, alpha=0.7, label=r"Theory $B_2(T, \epsilon)$")
ax.axhline(y=B2_hs, color="orange", ls="--", lw=1, label=f"Hard sphere B₂={B2_hs:.5f}")
ax.axhline(y=0, color="gray", ls="-", lw=0.5, alpha=0.5)
ax.set_xlabel(r"Lennard-Jones $\epsilon$", fontsize=14)
ax.set_ylabel(r"$B_2^{eff} = (Z-1)/\rho$", fontsize=14)
ax.set_title("Effective Second Virial Coefficient (ε ≤ 2.5)", fontsize=13)
ax.legend(fontsize=10, loc="best")
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.1, 2.6)
ax.set_ylim(-0.2, 0.03)

plt.tight_layout()
plt.savefig("eos_analysis/eps_fine_zoomed.png", dpi=150, bbox_inches="tight")
print("Saved zoomed plot")
