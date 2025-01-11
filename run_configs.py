import subprocess
import toml
import os
import shutil
import asyncio

def create_config_file(config, file_path):
    with open(file_path, 'w') as f:
        toml.dump(config, f)

async def run_rust_program(config_path, n_step, bounded):
    with open(config_path+'.log', 'w') as log:
        process = await asyncio.create_subprocess_exec(
            'cargo', 'run', '--release', '--', config_path, str(n_step), 'true' if bounded else 'false',
            stdout=log, stderr=log
        )
        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            print(f'Finished running {config_path}')
        else:
            print(f'Error running {config_path}: {stderr.decode()},\nstdout:{stdout.decode()}, check {config_path}.log for more info')

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
    } for D in [0.11, 0.13, 0.14, 0.16, 0.18, 0.19]
]

async def main():
    tasks = []
    for i, config in enumerate(configs):
        config_path = f'config_D{config['D']}.toml'
        if not os.path.exists(config_path):
            create_config_file(config, config_path)
            task = asyncio.create_task(run_rust_program(config_path, 500, True))
            tasks.append(task)
        print(f'Start running config_D{config["D"]}')
        await asyncio.sleep(1)
    await asyncio.gather(*tasks)
    for config in configs:
        os.remove(f'config_D{config["D"]}.toml')

asyncio.run(main())
for d in os.listdir('./data'):
    if os.path.isdir(f'./data/{d}') and 'plot.py' not in os.listdir(f'./data/{d}'):
        shutil.copy('./data/N=131072_L=16_D=0.1_T=1_MASS=200_N_TEST=1/plot.py', f'./data/{d}')
    if os.path.isdir(f'./data/{d}') and not os.path.exists(f'./data/{d}/pressure.png'):
        subprocess.run(['python', 'plot.py'], cwd=f'./data/{d}')
