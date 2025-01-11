import subprocess
import toml
import os
import shutil

def create_config_file(config, file_path):
    with open(file_path, 'w') as f:
        toml.dump(config, f)

def run_rust_program(config_path, n_step, bounded):
    subprocess.run(['cargo', 'run', '--release', '--', config_path, str(n_step), 'true' if bounded else 'false'])

N = 131072
L = 16
T = 1.0
MASS = 200.0
N_TEST = 1
configs = [
    {
        'N': N,
        'L': L,
        'D': D,
        'T': T,
        'MASS': MASS,
        'N_TEST': N_TEST,
        'E0': 1.5 * T,
        'T_STEP': float(f"{0.002 / D ** 2:.2f}"),
    } for D in [0.1, 0.12, 0.15, 0.17, 0.2]
]

for i, config in enumerate(configs):
    config_path = f'config_{i}.toml'
    if not os.path.exists(config_path):
        create_config_file(config, config_path)
        run_rust_program(config_path, 500, True)
    print(f'Finished running config {i}')
    os.remove(config_path)
for d in os.listdir('./data'):
    if os.path.isdir(f'./data/{d}') and 'plot.py' not in os.listdir(f'./data/{d}'):
        shutil.copy('./data/N=131072_L=16_D=0.1_T=1_MASS=200_N_TEST=1/plot.py', f'./data/{d}')
    if os.path.isdir(f'./data/{d}') and not os.path.exists(f'./data/{d}/pressure.png'):
        subprocess.run(['python', 'plot.py'], cwd=f'./data/{d}')
