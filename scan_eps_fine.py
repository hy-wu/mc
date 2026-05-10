#!/usr/bin/env python3
"""
Fine-grained epsilon scan with theoretical comparison.
- Wider range: 0 to 50
- Finer granularity: ~20 points on log scale
- Compare with LJ second virial coefficient B2(T*)
"""
import subprocess
import os
import json
import numpy as np

def rust_fmt(x):
    if x == int(x):
        return str(int(x))
    return str(x)

# ============================================================
# Parameters
# ============================================================
N = 16384
L = 8
MASS = 200.0
N_TEST = 1
T = 1.0
E0 = 1.5 * T
BOUNDED = True
N_STEP = 5000  # More steps for better statistics
d = 0.2
dt = 0.25  # 0.01 / d^2

BINARY = "./target/release/mc"

# Fine-grained epsilon values (log-spaced + linear near 0)
eps_values = sorted(set([
    0.0,
    0.001, 0.002, 0.005,
    0.01, 0.02, 0.05,
    0.1, 0.2, 0.3, 0.5, 0.7,
    1.0, 1.5, 2.0, 3.0, 5.0,
    7.0, 10.0, 15.0, 20.0, 30.0, 50.0,
]))

configs = []
for eps in eps_values:
    configs.append({
        "n": N, "l": L, "d": d,
        "temperature": T, "mass": MASS, "n_test": N_TEST,
        "e0": E0, "dt": dt, "l_j_epsilon": eps,
        "label": f"eps_{eps}",
    })

def write_toml(cfg, path):
    with open(path, "w") as f:
        for k in ["n", "l", "d", "temperature", "mass", "n_test", "e0", "dt", "l_j_epsilon"]:
            v = cfg[k]
            f.write(f"{k} = {v}\n")

def run_one(cfg):
    label = cfg["label"]
    config_path = f"scan_configs/{label}.toml"
    os.makedirs("scan_configs", exist_ok=True)
    write_toml(cfg, config_path)

    # Check if data already exists
    eps = cfg["l_j_epsilon"]
    data_dir = f"data/N={N}_L={L}_D={d}_T={rust_fmt(T)}_MASS={int(MASS)}_N_TEST={N_TEST}_T_STEP={rust_fmt(dt)}_EPS={rust_fmt(eps)}_N_STEP={N_STEP}_bounded=true"
    if os.path.exists(os.path.join(data_dir, "pressure.csv")):
        print(f"  {label}: cached", flush=True)
        return label

    print(f"  Running {label} (ε={eps}, dt={dt}) ...", flush=True)
    result = subprocess.run(
        [BINARY, config_path, str(N_STEP), "true"],
        capture_output=True, text=True, timeout=1200
    )
    if result.returncode != 0:
        print(f"  ERROR in {label}: {result.stderr[:200]}")
        return None

    for line in result.stdout.strip().split("\n"):
        if "ms per step" in line:
            ms = int(line.split()[0])
            print(f"  {label}: {ms} ms/step")
    return label

def main():
    print("Building release binary...")
    subprocess.run(["cargo", "build", "--release"], check=True)
    print(f"Running {len(configs)} epsilon configurations, {N_STEP} steps each")
    print("=" * 60)

    results = []
    for i, cfg in enumerate(configs):
        print(f"[{i+1}/{len(configs)}]", end="")
        label = run_one(cfg)
        if label:
            results.append(cfg)

    # Save metadata
    meta = {
        "N": N, "L": L, "T": T, "MASS": MASS, "N_STEP": N_STEP,
        "BOUNDED": BOUNDED, "d": d, "dt": dt, "configs": [],
    }
    for cfg in results:
        eps = cfg["l_j_epsilon"]
        meta["configs"].append({
            "label": cfg["label"],
            "l_j_epsilon": eps,
            "data_dir": f"data/N={N}_L={L}_D={d}_T={rust_fmt(T)}_MASS={int(MASS)}_N_TEST={N_TEST}_T_STEP={rust_fmt(dt)}_EPS={rust_fmt(eps)}_N_STEP={N_STEP}_bounded=true",
        })
    with open("scan_eps_fine.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. Run: python3 analyze_eps_fine.py")

if __name__ == "__main__":
    main()
