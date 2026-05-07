import os
import math
import random
import subprocess
import multiprocessing
from datetime import datetime

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
SAVE_ROOT_DEFAULT = os.path.join(REPO_ROOT, "outputs", "experiment", "v250312")

cfg = {
    "v01": {
        "duration": 12,
        "s": 500,
        "S": 500,
        "h": [80, 10],
        "H": [0, 1],
        "a": [0, 0.5],
        "v": [10],
        "V": [10],
        "f": [10],
        "F": [10],
        "q": [1],
    },
    "v02": {
        "duration": 12,
        "s": 500,
        "S": 500,
        "h": [80, 10],
        "H": [0, 1],
        "a": [1, 1],
        "f": [5],
        "F": [2],
        "v": [25],
        "V": [25],
        "q": [1],
    },

}

save_root = SAVE_ROOT_DEFAULT
timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
save_dir = os.path.join(save_root, "simulator", "ecgsyn", timestamp)
os.makedirs(save_dir, exist_ok=True)

random.seed(42)

def rand_bell(a, b):
    u1 = random.uniform(0, 1)
    u2 = random.uniform(0, 1)
    
    z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return z0 * b + a

def gen_params(param_ver: str):
    sim_cfg = cfg[param_ver]
    s = sim_cfg["s"]
    S = sim_cfg["S"]
    duration = sim_cfg["duration"]

    h = rand_bell(sim_cfg["h"][0], sim_cfg["h"][1])
    H = max(0, rand_bell(sim_cfg["H"][0], sim_cfg["H"][1])) + 1
    a = max(0, rand_bell(sim_cfg["a"][0], sim_cfg["a"][1]))
    n = int(h / 60 * duration + 5) # + 5 as a buffer to avoid too short simulation
    v = random.uniform(0, 1) / sim_cfg["v"][0]
    V = random.uniform(0, 1) / sim_cfg["V"][0]
    f = random.uniform(0, 1) / sim_cfg["f"][0]
    F = random.uniform(0, 1) / sim_cfg["F"][0]
    q = random.uniform(0, 1) / sim_cfg["q"][0]
    R = random.uniform(0, 1000)
    command_param = f"-n {n} -s {s} -S {S} -h {h} -H {H} -a {a} -v {v} -V {V} -f {f} -F {F} -q {q} -R {R}"
    return command_param

# def run_simulator(idx, duration=12):

#     command_param = gen_params(duration)
#     gen_command = f"./ecgsyn {command_param}"
#     os.system(gen_command)
#     dirname = os.path.join(
#         save_dir,
#         "dat_files",
#         f"id_{idx:08d}"
#     )
#     os.makedirs(dirname, exist_ok=True)
#     mv_command = f"mv *.dat {dirname}/"
#     os.system(mv_command)


def run_simulator(idx, save_dir, param_ver):
    """Runs the ecgsyn simulator and moves the output."""
    command_param = gen_params(param_ver)
    gen_command = f"./ecgsyn {command_param} -O syn{idx:08d}.dat"

    # Use subprocess.run for better control and error handling
    result = subprocess.run(gen_command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running ecgsyn for idx {idx}: {result.stderr}")
        return

    dirname = os.path.join(
        save_dir, 
        "dat_files", 
        f"id{idx//1000+1:04d}",
        f"id{idx:08d}"
    )
    os.makedirs(dirname, exist_ok=True)
    mv_command = f"mv syn{idx:08d}.dat {dirname}/"

    # Use subprocess.run for mv command as well
    mv_result = subprocess.run(mv_command, shell=True, capture_output=True, text=True)

    if mv_result.returncode != 0:
        print(f"Error moving files for idx {idx}: {mv_result.stderr}")
        return

def run_simulations_parallel(num_simulations, save_dir, param_ver, num_processes=None, skip=None):
    """Runs simulations in parallel using multiprocessing."""

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pool = multiprocessing.Pool(processes=num_processes)
    if skip is None:
        tasks = [(idx, save_dir, param_ver) for idx in range(num_simulations)]
    else:
        tasks = [(idx, save_dir, param_ver) for idx in range(skip, num_simulations)]


    results = []
    for t in tasks:
        results.append(pool.apply_async(run_simulator, t))

    for i, res in enumerate(results):
        res.get()  # Wait for each task to complete
        if i % 5000 == 0:
            print(f"Completed {i+1}/{num_simulations} simulations.")
        # print(f"Simulation {i+1}/{num_simulations} completed.")

    pool.close()
    pool.join()

if __name__ == "__main__":
    # n_gen = 1200000
    # for idx in range(n_gen):
    #     if idx % 10000 == 0:
    #         print(f"Generating {idx+1}/{n_gen}")
    #     run_simulator(idx+1)

    num_simulations = 1200000  # Example number of simulations
    n_proc = 60  # Number of processes to use
    param_ver = "v02"
    skip = 717395
    run_simulations_parallel(
        num_simulations, 
        save_dir, 
        param_ver, 
        num_processes=n_proc,
        skip=skip
    )    
