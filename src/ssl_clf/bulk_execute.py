import os
from argparse import ArgumentParser

import yaml

from experiment import ExperimentManager, DemographicsExperimentManager


def split_exp_targets(exp_targets):
    if exp_targets.isdigit():
        exp_ids = [int(exp_targets)]
    elif exp_targets.find(",") != -1:
        exp_ids = [int(v) for v in exp_targets.split(",")]
    elif exp_targets.find("-") != -1:
        s_e = exp_targets.split("-")
        s, e = int(s_e[0]), int(s_e[-1])
        exp_ids = [i for i in range(s, e+1)]
    else:
        raise
    return exp_ids

parser = ArgumentParser()

parser.add_argument(
    '--ids', 
    default="9001,9002"
)
parser.add_argument(
    '--device', 
    default="cuda:0"
)
parser.add_argument(
    '--show_error', 
    action="store_true"
)

args = parser.parse_args()

def check_yaml_exist(exp_id):
    exp_config_file = os.path.join(
        "./resources",
        f"exp{exp_id//100:02d}s",
        f"exp{exp_id:04d}.yaml"
    )
    return os.path.exists(exp_config_file)

def check_eval_only(exp_id):
    exp_config_file = os.path.join(
        "./resources",
        f"exp{exp_id//100:02d}s",
        f"exp{exp_id:04d}.yaml"
    )
    with open(exp_config_file, "r") as fp:
        config = yaml.safe_load(fp)
    
    if "eval_demographics" not in config:
        return False
    return config["eval_demographics"]["param_val"]

errors = []
ids = split_exp_targets(args.ids)
for exe_id in ids:
    
    if not check_yaml_exist(int(exe_id)):
        errors.append([int(exe_id), args.device])
        continue

    if check_eval_only(int(exe_id)):
        executer = DemographicsExperimentManager(
            int(exe_id), 
            args.device,
            debug=False
        )
    else:
        executer = ExperimentManager(
            int(exe_id), 
            args.device,
            debug=False
        )

    if args.show_error:
        executer.main(single_run=False)
    else:
        try:
            executer.main(single_run=False)
        except:
            errors.append([int(exe_id), args.device])
print("*"*80)
for e in errors:
    print(e)

