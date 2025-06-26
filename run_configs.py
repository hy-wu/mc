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

N = 16384
L = 8
T = 1.0
MASS = 200.0
N_TEST = 1
configs = [
    # {
    #     'n': N,
    #     'l': L,
    #     'd': D,
    #     'temperature': T,
    #     'mass': MASS,
    #     'n_test': N_TEST,
    #     'e0': 1.5 * T,
    #     'dt': float(f"{0.002 / D ** 2:.2f}"),
    # # } for D in [0.10]
    # } for D in [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]
    {
        'n': 16384,
        'l': 8,
        'd': 0.2,
        'temperature': T,
        'mass': 200.0,
        'n_test': 1,
        'e0': 1.5 * T,
        'dt': float(f"{0.002 / 0.2 ** 2 / T ** 0.5:.2g}"),
    # } for D in [0.10]
    # } for T in [0.01, 0.1, 1, 10, 100, 1000]
    } for T in [30, 100, 300, 1000]
]

async def main():
    tasks = []
    for i, config in enumerate(configs):
        config_path = f'config_D{config["d"]}_T{config["temperature"]}.toml'
        # if not os.path.exists(config_path):
        create_config_file(config, config_path)
        task = asyncio.create_task(run_rust_program(config_path, 10000, True))
        tasks.append(task)
        print(f'Start running config_D{config["d"]}_T{config["temperature"]}')
        await asyncio.sleep(1)
    await asyncio.gather(*tasks)
    for config in configs:
        os.remove(f'config_D{config["d"]}_T{config["temperature"]}.toml')

asyncio.run(main())
for d in os.listdir('./data'):
    if os.path.isdir(f'./data/{d}') and 'plot.py' not in os.listdir(f'./data/{d}'):
        shutil.copy('./data/plot.py', f'./data/{d}')
    if os.path.isdir(f'./data/{d}') and not os.path.exists(f'./data/{d}/pressure.png'):
        subprocess.run(['python', 'plot.py'], cwd=f'./data/{d}')
