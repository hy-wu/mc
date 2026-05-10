#!/usr/bin/env python3
"""
Parameter scan for EOS verification.
Scan 1: d sweep with l_j_epsilon=0 (pure hard sphere → verify Carnahan-Starling)
Scan 2: l_j_epsilon sweep with fixed d=0.2 (LJ effects)
"""
import subprocess
import os
import sys
import math
import json

def rust_fmt(x):
    """Mimic Rust's f64 Display formatting: 1.0 → '1', 0.25 → '0.25'"""
    if x == int(x):
        return str(int(x))
    return str(x)

# ============================================================
# Fixed parameters
# ============================================================
N = 16384
L = 8
MASS = 200.0
N_TEST = 1
T = 1.0
E0 = 1.5 * T
BOUNDED = True
N_STEP = 3000  # 2000 relaxation + 1000 sampling

BINARY = "./target/release/mc"

# ============================================================
# Scan definitions
# ============================================================

def make_dt(d, T):
    """Scale dt so collision probability stays < 1."""
    return round(0.01 / (d ** 2 * T ** 0.5), 4)

# Scan 1: d sweep, pure hard sphere (eps=0)
scan1_d_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]
scan1_configs = []
for d in scan1_d_values:
    scan1_configs.append({
        "n": N, "l": L, "d": d,
        "temperature": T, "mass": MASS, "n_test": N_TEST,
        "e0": E0, "dt": make_dt(d, T), "l_j_epsilon": 0.0,
        "label": f"scan1_d{d}",
    })

# Scan 2: l_j_epsilon sweep, fixed d=0.2
scan2_eps_values = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
scan2_configs = []
for eps in scan2_eps_values:
    d = 0.2
    scan2_configs.append({
        "n": N, "l": L, "d": d,
        "temperature": T, "mass": MASS, "n_test": N_TEST,
        "e0": E0, "dt": make_dt(d, T), "l_j_epsilon": eps,
        "label": f"scan2_eps{eps}",
    })

all_configs = scan1_configs + scan2_configs

# ============================================================
# Run simulations
# ============================================================

def write_toml(cfg, path):
    with open(path, "w") as f:
        for k in ["n", "l", "d", "temperature", "mass", "n_test", "e0", "dt", "l_j_epsilon"]:
            v = cfg[k]
            if isinstance(v, float):
                f.write(f"{k} = {v}\n")
            else:
                f.write(f"{k} = {v}\n")

def run_one(cfg):
    label = cfg["label"]
    config_path = f"scan_configs/{label}.toml"
    os.makedirs("scan_configs", exist_ok=True)
    write_toml(cfg, config_path)

    bounded_str = "true" if BOUNDED else "false"
    print(f"  Running {label} (d={cfg['d']}, eps={cfg['l_j_epsilon']}, dt={cfg['dt']}) ...", flush=True)
    result = subprocess.run(
        [BINARY, config_path, str(N_STEP), bounded_str],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        print(f"  ERROR in {label}: {result.stderr[:200]}")
        return None

    # Extract timing
    for line in result.stdout.strip().split("\n"):
        if "ms per step" in line:
            ms = int(line.split()[0])
            print(f"  {label}: {ms} ms/step")

    return label

def main():
    # Build first
    print("Building release binary...")
    subprocess.run(["cargo", "build", "--release"], check=True)
    print(f"Running {len(all_configs)} configurations, {N_STEP} steps each")
    print("=" * 60)

    results = []
    for i, cfg in enumerate(all_configs):
        print(f"[{i+1}/{len(all_configs)}]", end="")
        label = run_one(cfg)
        if label:
            results.append(cfg)

    # Save scan metadata for analysis
    meta = {
        "N": N, "L": L, "T": T, "MASS": MASS, "N_STEP": N_STEP,
        "BOUNDED": BOUNDED, "configs": [],
    }
    for cfg in results:
        meta["configs"].append({
            "label": cfg["label"],
            "d": cfg["d"],
            "l_j_epsilon": cfg["l_j_epsilon"],
            "dt": cfg["dt"],
            "data_dir": f"data/N={N}_L={L}_D={cfg['d']}_T={rust_fmt(T)}_MASS={int(MASS)}_N_TEST={N_TEST}_T_STEP={rust_fmt(cfg['dt'])}_EPS={rust_fmt(cfg['l_j_epsilon'])}_N_STEP={N_STEP}_bounded={'true' if BOUNDED else 'false'}",
        })
    with open("scan_results.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nAll done. Metadata saved to scan_results.json")
    print(f"Run: python3 analyze_eos.py")

if __name__ == "__main__":
    main()
