import os
import socket
from argparse import Namespace
from glob import glob

import yaml
import torch
import numpy as np

from codes.generator import Generator
from experiment import ExperimentManagerBase
from codes.utils import utils

torch.backends.cudnn.deterministic = True

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)

def run_generate(
    args,
    weight_file_loc: str,
    save_root: str,
    generate_train_data: bool
):
    """
    Execute train code for ecg classifier
    Args:
        args (Namespace): Namespace for parameters used.
        save_root (str): 
    Returns:
        best_val_loss (float): 
        save_dir (str):
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if generate_train_data:
        save_setting = "train"
        args.n_total_samples = args.n_total_samples_train
    else:
        save_setting = "val"
        args.n_total_samples = args.n_total_samples_val
    
    # Prepare result storing directories
    save_dir = os.path.join(
        save_root, 
        save_setting
    )
    generator = Generator(args, save_dir)
    generator.set_model()
    generator.set_weight(weight_file_loc)
    generator.run()

    return save_dir

class SampleGenerationManager(ExperimentManagerBase):

    def __init__(self, gen_id: int, device: str, debug: bool=False) -> None:
        """
        Args:
            args (Namespace):
        Returns:
            None
        """
        gen_config_file = os.path.join(
            config["experiment"]["path"]["gen_yaml_loc"],
            f"g{gen_id//100:02d}s",
            f"gen{gen_id:04d}.yaml"
        )

        self._prepare_fixed_params(gen_config_file, device)
        self._prepare_save_loc(gen_id)

    def _prepare_fixed_params(self, exp_config_file, device):

        fixed_params =\
            self._load_train_params(exp_config_file)
        # fixed_params = self._update_fixed_params(fixed_params)
        # fixed_params = self._insert_freq_to_fixed_param(fixed_params)
        fixed_params = self._merge_setting(fixed_params)
        fixed_params = Namespace(**fixed_params)

        fixed_params.device = device
        fixed_params.seed = config["experiment"]["seed"]["generate"]
        fixed_params.host = socket.gethostname()
        fixed_params.is_gan = fixed_params.dgm in config["generatives"]["gans"]
        fixed_params.batch_size = min(
            fixed_params.batch_size, fixed_params.n_per_file)
        self.fixed_params = fixed_params

    def _get_src_model_loc(self, key):
        model = key.split("/")[0]
        weight_id = key.split("/")[1]
        return config["generatives"]["model_path"][model][weight_id]

    def _merge_setting(self, fixed_params):
        """
        Load yaml file used during mae pretraining.

        Args:
            fixed_params (dict): 
        Returns:
            fixed_params (dict): 
        """
        src_model_loc =\
            self._get_src_model_loc(fixed_params["weight_file_key"])
        yaml_file = glob(src_model_loc + "/exp_config.yaml")
        assert len(yaml_file) == 1

        with open(yaml_file[0]) as f:
            model_cfg = yaml.safe_load(f)
        fixed_params = self._update_params(fixed_params, model_cfg)
        return fixed_params

    def _prepare_save_loc(self, gen_id: int):
        """
        Args:

        Returns:
            None
        """
        self.save_loc = os.path.join(
            config["experiment"]["path"]["save_root_gen"],
            f"gen{gen_id:04d}"[:-2]+"s",
            f"gen{gen_id:04d}",
        )
        os.makedirs(self.save_loc, exist_ok=True)

    def main(self):

        weight_file_dir = self._get_src_model_loc(
            self.fixed_params.weight_file_key)
        
        timestamp = utils.get_timestamp()
        save_setting = f"{timestamp}-{self.fixed_params.host}"
        save_dir = os.path.join(
            self.save_loc, 
            save_setting
        )

        run_generate(
            self.fixed_params,
            weight_file_dir,
            save_dir,
            generate_train_data=True
        )
        run_generate(
            self.fixed_params,
            weight_file_dir,
            save_dir,
            generate_train_data=False
        )


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        '--exp', 
        default=0
    )
    parser.add_argument(
        '--device', 
        default="cuda:0"
    )
    parser.add_argument(
        '--debug', 
        action="store_true"
    )
    args = parser.parse_args()

    print(args)

    executer = SampleGenerationManager(
        int(args.exp), 
        args.device,
        debug=args.debug
    )
    executer.main()
