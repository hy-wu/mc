#!/usr/bin/env python3
"""
Analyze fine-grained epsilon scan results.
Compare with LJ second virial coefficient theory.
"""
import json
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import integrate

plt.rcParams.update({
    "font.size": 12,
    "figure.dpi": 150,
    "font.family": "sans-serif",
})

# ============================================================
# LJ Theory: Second Virial Coefficient B2(T*)
# ============================================================

def lj_potential(r, sigma, epsilon):
    """Standard 12-6 Lennard-Jones potential u(r)."""
    sr6 = (sigma / r) ** 6
    return 4.0 * epsilon * (sr6**2 - sr6)

def compute_B2_lj(T, sigma, epsilon, r_min=0.01, r_max_factor=10.0):
    """
    Numerically compute the second virial coefficient B2 for LJ potential.
    B2 = -2π ∫_0^∞ [exp(-u(r)/kT) - 1] r² dr
    with kB = 1.
    Uses the same cutoff as the simulation: r_max = sqrt(10) * sigma.
    """
    r_max = math.sqrt(r_max_factor) * sigma  # match simulation cutoff
    
    def integrand(r):
        u = lj_potential(r, sigma, epsilon)
        return (math.exp(-u / T) - 1.0) * r**2
    
    # Split integral at sigma to handle the steep repulsive wall
    # For very small r, exp(-u/kT) ≈ 0 so integrand ≈ -r²
    r_inner = max(r_min * sigma, 0.3 * sigma)
    result1, _ = integrate.quad(integrand, r_inner, sigma, limit=200)
    result2, _ = integrate.quad(integrand, sigma, r_max, limit=200)
    
    B2 = -2.0 * math.pi * (result1 + result2)
    return B2

def compute_B2_hard_sphere(sigma):
    """B2 for hard spheres = 2π σ³ / 3."""
    return 2.0 * math.pi * sigma**3 / 3.0

# ============================================================
# van der Waals approximation
# ============================================================
def vdw_Z(rho, T, b, a):
    """van der Waals EOS: Z = V/(V-Nb) - Na/(VkT) = 1/(1-bρ) - aρ/kT"""
    return 1.0 / (1.0 - b * rho) - a * rho / T

def lj_vdw_params(sigma, epsilon):
    """
    Approximate vdW parameters from LJ potential:
    b = 2π σ³ / 3 (excluded volume)
    a = -∫_{σ}^{∞} u_att(r) 4πr² dr  (attraction integral)
    For LJ with cutoff at sqrt(10)*σ:
    """
    b = 2.0 * math.pi * sigma**3 / 3.0
    # Numerically integrate the attractive part
    r_max = math.sqrt(10) * sigma
    def integrand(r):
        u = lj_potential(r, sigma, epsilon)
        if u < 0:
            return -u * 4 * math.pi * r**2
        return 0.0
    a, _ = integrate.quad(integrand, sigma, r_max, limit=100)
    return b, a

# ============================================================
# Load and analyze
# ============================================================

def main():
    with open("scan_eps_fine.json") as f:
        meta = json.load(f)

    N = meta["N"]
    L = meta["L"]
    T = meta["T"]
    d = meta["d"]
    rho = N / L**3
    sigma = d  # particle diameter = LJ sigma

    output_dir = "eos_analysis"
    os.makedirs(output_dir, exist_ok=True)

    normal_eps_threshold = 1.0

    # Load simulation data
    sim_data = []
    for c in meta["configs"]:
        if c["l_j_epsilon"] > normal_eps_threshold:
            continue
        else:
            pressure_path = os.path.join(c["data_dir"], "pressure.csv")
            if not os.path.exists(pressure_path):
                print(f"  MISSING: {c['label']}")
                continue
            df = pd.read_csv(pressure_path)
            p = df["pressure"].values
            # Use last 60% for equilibrium average
            n_relax = len(p) * 2 // 5
            p_eq = p[n_relax:]
            p_mean = np.mean(p_eq)
            p_sem = np.std(p_eq) / np.sqrt(len(p_eq))
            eps = c["l_j_epsilon"]
            Z = p_mean / (rho * T)
            Z_sem = p_sem / (rho * T)
            sim_data.append({"eps": eps, "P": p_mean, "Z": Z, "Z_sem": Z_sem})
            print(f"  ε={eps:8.3f}  P={p_mean:8.3f}  Z={Z:.5f} ± {Z_sem:.5f}")

    if not sim_data:
        print("No data found!")
        return

    eps_arr = np.array([x["eps"] for x in sim_data])
    Z_arr = np.array([x["Z"] for x in sim_data])
    Z_sem_arr = np.array([x["Z_sem"] for x in sim_data])
    Z_baseline = sim_data[0]["Z"]  # eps=0 baseline

    # ================================================================
    # Compute theoretical curves
    # ================================================================

    # 1. Second virial coefficient theory: Z = 1 + B2(T*) * ρ
    eps_theory = np.linspace(0.001, 55, 500)
    B2_theory = np.array([compute_B2_lj(T, sigma, e) for e in eps_theory])
    B2_hs = compute_B2_hard_sphere(sigma)
    Z_virial = 1.0 + B2_theory * rho  # First-order virial expansion

    # 2. van der Waals approximation
    Z_vdw = []
    for e in eps_theory:
        b, a = lj_vdw_params(sigma, e)
        Z_vdw.append(vdw_Z(rho, T, b, a))
    Z_vdw = np.array(Z_vdw)

    # 3. B2 for each simulation epsilon
    B2_sim_theory = np.array([compute_B2_lj(T, sigma, e) if e > 0 else B2_hs for e in eps_arr])
    Z_virial_at_sim = 1.0 + B2_sim_theory * rho

    # ================================================================
    # PLOT 1: Z vs ε — main comparison
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Z vs ε (linear scale for ε)
    ax = axes[0, 0]
    ax.errorbar(eps_arr, Z_arr, yerr=Z_sem_arr, fmt="ro-", ms=6, capsize=3, lw=1.5,
                label="Simulation", zorder=5)
    ax.plot(eps_theory, Z_virial, "b--", lw=2, alpha=0.7,
            label=r"Virial: $Z = 1 + B_2(\epsilon)\rho$")
    ax.plot(eps_theory, Z_vdw, "g:", lw=2, alpha=0.7,
            label="van der Waals approx")
    ax.axhline(y=1.0, color="gray", ls="-", lw=0.5, alpha=0.5)
    ax.set_xlabel(r"Lennard-Jones $\epsilon$", fontsize=14)
    ax.set_ylabel("Compressibility factor Z", fontsize=14)
    ax.set_title("Z vs ε (linear scale)", fontsize=13)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xlim(np.min(eps_arr) * 0.9, np.max(eps_arr) * 1.1)
    ax.set_ylim(np.min(Z_arr) * 0.9, np.max(Z_arr) * 1.1)

    # Panel 2: Z vs ε (log scale for ε)
    ax = axes[0, 1]
    ax.errorbar(eps_arr[1:], Z_arr[1:], yerr=Z_sem_arr[1:], fmt="ro-", ms=6, capsize=3, lw=1.5,
                label="Simulation", zorder=5)
    ax.plot(eps_theory, Z_virial, "b--", lw=2, alpha=0.7,
            label=r"Virial: $Z = 1 + B_2(\epsilon)\rho$")
    ax.plot(eps_theory, Z_vdw, "g:", lw=2, alpha=0.7,
            label="van der Waals approx")
    ax.axhline(y=Z_baseline, color="orange", ls="--", lw=1,
               label=f"Hard sphere Z={Z_baseline:.3f}")
    ax.set_xlabel(r"Lennard-Jones $\epsilon$", fontsize=14)
    ax.set_ylabel("Compressibility factor Z", fontsize=14)
    ax.set_title("Z vs ε (log scale)", fontsize=13)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(np.min(eps_arr) * 0.9, np.max(eps_arr) * 1.1)
    ax.set_ylim(np.min(Z_arr) * 0.9, np.max(Z_arr) * 1.1)

    # Panel 3: ΔZ = Z(ε) - Z(0) — the LJ correction
    ax = axes[1, 0]
    dZ_sim = Z_arr - Z_baseline
    dZ_virial = Z_virial - (1.0 + B2_hs * rho)  # theoretical ΔZ
    dZ_vdw = Z_vdw - Z_vdw[0] if len(Z_vdw) > 0 else Z_vdw

    ax.errorbar(eps_arr, dZ_sim, yerr=Z_sem_arr, fmt="ro-", ms=6, capsize=3, lw=1.5,
                label="Simulation ΔZ")
    ax.plot(eps_theory, dZ_virial, "b--", lw=2, alpha=0.7,
            label=r"Virial $\Delta Z = [B_2(\epsilon) - B_2^{HS}]\rho$")
    ax.axhline(y=0, color="gray", ls="-", lw=0.5)
    ax.set_xlabel(r"Lennard-Jones $\epsilon$", fontsize=14)
    ax.set_ylabel("ΔZ = Z(ε) - Z(0)", fontsize=14)
    ax.set_title("LJ correction to EOS", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(np.min(eps_arr) - np.abs(np.min(eps_arr))*0.1, np.max(eps_arr) * 1.1)
    ax.set_ylim(np.min(dZ_sim) - np.abs(np.min(dZ_sim))*0.1, np.max(dZ_sim) * 1.1)

    # Panel 4: B2 effective — extract simulation's effective B2
    ax = axes[1, 1]
    # From Z = 1 + B2_eff * ρ, so B2_eff = (Z-1)/ρ
    B2_eff_sim = (Z_arr - 1.0) / rho
    B2_eff_sem = Z_sem_arr / rho

    ax.errorbar(eps_arr, B2_eff_sim, yerr=B2_eff_sem, fmt="ro-", ms=6, capsize=3, lw=1.5,
                label=r"Simulation $B_2^{eff}$", zorder=5)
    ax.plot(eps_theory, B2_theory, "b--", lw=2, alpha=0.7,
            label=r"Theory $B_2(T, \epsilon)$")
    ax.axhline(y=B2_hs, color="orange", ls="--", lw=1,
               label=f"Hard sphere B₂={B2_hs:.5f}")
    ax.set_xlabel(r"Lennard-Jones $\epsilon$", fontsize=14)
    ax.set_ylabel(r"$B_2^{eff} = (Z-1)/\rho$", fontsize=14)
    ax.set_title("Effective second virial coefficient", fontsize=13)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xlim(np.min(eps_arr) - np.abs(np.min(eps_arr))*0.1, np.max(eps_arr) * 1.1)
    ax.set_ylim(np.min(B2_eff_sim) - np.abs(np.min(B2_eff_sim))*0.1, np.max(B2_eff_sim) * 1.1)

    plt.suptitle(f"LJ EOS Analysis: d={d}, T={T}, ρ={rho}, N={N}", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eps_fine_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {output_dir}/eps_fine_analysis.png")

    # ================================================================
    # PLOT 2: Pressure time series for selected epsilon values
    # ================================================================
    selected_eps = [0.0, 0.1, 1.0, 5.0, 20.0, 50.0]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()
    
    plot_idx = 0
    for c in meta["configs"]:
        if c["l_j_epsilon"] not in selected_eps:
            continue
        pressure_path = os.path.join(c["data_dir"], "pressure.csv")
        if not os.path.exists(pressure_path):
            continue
        if plot_idx >= 6:
            break
        df = pd.read_csv(pressure_path)
        ax = axes_flat[plot_idx]
        ax.plot(df["time"], df["pressure"], alpha=0.2, lw=0.5, color="blue")
        rolled = df["pressure"].rolling(window=100).mean()
        ax.plot(df["time"], rolled, color="red", lw=1.5, label="MA-100")
        n_relax = len(df) * 2 // 5
        ax.axvline(x=df["time"].iloc[n_relax], color="green", ls="--", alpha=0.7, label="Relax end")
        ax.set_title(f"ε = {c['l_j_epsilon']}", fontsize=12)
        ax.set_xlabel("Time")
        ax.set_ylabel("Pressure")
        ax.legend(fontsize=8)
        plot_idx += 1

    plt.suptitle("Pressure Equilibration at Different ε", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eps_fine_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/eps_fine_timeseries.png")

    # ================================================================
    # PLOT 3: Phase diagram — Z vs T* = kT/ε
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # T* = kT/ε (reduced temperature)
    mask = eps_arr > 0
    Tstar_sim = T / eps_arr[mask]
    Z_sim_nz = Z_arr[mask]
    Z_sem_nz = Z_sem_arr[mask]
    
    ax.errorbar(Tstar_sim, Z_sim_nz, yerr=Z_sem_nz, fmt="ro", ms=8, capsize=4, lw=1.5,
                label="Simulation", zorder=5)
    
    # Theory curve
    Tstar_theory = T / eps_theory
    ax.plot(Tstar_theory, Z_virial, "b--", lw=2, alpha=0.7,
            label=r"Virial: $Z = 1 + B_2(T^*)\rho$")
    
    # Boyle temperature: B2(T_B) = 0 → Z = 1
    # Find it numerically
    from scipy.optimize import brentq
    try:
        def b2_zero(eps_val):
            return compute_B2_lj(T, sigma, eps_val)
        eps_boyle = brentq(b2_zero, 0.001, 100)
        Tstar_boyle = T / eps_boyle
        ax.axvline(x=Tstar_boyle, color="purple", ls=":", lw=2, alpha=0.7,
                   label=f"Boyle T* = {Tstar_boyle:.2f}")
    except:
        pass
    
    ax.axhline(y=1.0, color="gray", ls="-", lw=0.5, alpha=0.5)
    ax.set_xlabel(r"Reduced temperature $T^* = k_BT/\epsilon$", fontsize=14)
    ax.set_ylabel("Compressibility factor Z", fontsize=14)
    ax.set_title(f"Z vs Reduced Temperature (d={d}, ρ={rho})", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(0.01, 1500)
    ax.set_ylim(np.min(Z_sim_nz) * 0.9, np.max(Z_sim_nz) * 1.1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eps_fine_Tstar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/eps_fine_Tstar.png")

    # ================================================================
    # PLOT 4: Pressure vs ε
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    P_arr = np.array([x["P"] for x in sim_data])
    P_sem_arr = np.array([x["Z_sem"] * rho * T for x in sim_data])
    
    ax.errorbar(eps_arr, P_arr, yerr=P_sem_arr, fmt="ro-", ms=8, capsize=4, lw=2, label="Simulation Pressure")
    ax.set_xlabel(r"Lennard-Jones $\epsilon$", fontsize=14)
    ax.set_ylabel("Pressure", fontsize=14)
    ax.set_xscale("log")
    ax.set_title(f"Pressure vs ε (d={d}, T={T}, ρ={rho})", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eps_fine_pressure.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/eps_fine_pressure.png")

    # ================================================================
    # Summary table
    # ================================================================
    print("\n" + "=" * 90)
    print(f"{'ε':>8s}  {'T*=kT/ε':>10s}  {'Z_sim':>10s}  {'Z_virial':>10s}  {'Z_vdW':>10s}  {'B2_eff':>10s}  {'B2_theory':>10s}")
    print("-" * 90)
    for i, sd in enumerate(sim_data):
        eps = sd["eps"]
        Tstar = T/eps if eps > 0 else float('inf')
        B2_eff = (sd["Z"] - 1.0) / rho
        B2_th = compute_B2_lj(T, sigma, eps) if eps > 0 else B2_hs
        Z_vir = 1.0 + B2_th * rho
        b, a = lj_vdw_params(sigma, eps) if eps > 0 else (B2_hs, 0)
        Z_vd = vdw_Z(rho, T, b, a)
        print(f"{eps:8.3f}  {Tstar:10.3f}  {sd['Z']:10.5f}  {Z_vir:10.5f}  {Z_vd:10.5f}  {B2_eff:10.6f}  {B2_th:10.6f}")
    print("=" * 90)

    # Save CSV
    df_out = pd.DataFrame(sim_data)
    df_out.to_csv(f"{output_dir}/eps_fine_results.csv", index=False)
    print(f"\nData saved to {output_dir}/eps_fine_results.csv")

if __name__ == "__main__":
    main()
