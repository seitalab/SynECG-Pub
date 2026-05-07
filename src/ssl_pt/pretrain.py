
import os
import socket
from argparse import Namespace

import yaml

from experiment import ExperimentManagerBase
from codes.run_train import run_train
from codes.supports.utils import get_timestamp

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)

class PretrainManager(ExperimentManagerBase):

    exp_mode = "pretrain"

    def __init__(
        self, 
        pretrain_id: int, 
        device: str, 
        use_cpu: bool=False,
        debug: bool=False
    ):
        pt_config_file = os.path.join(
            config["experiment"]["path"]["pretrain_yaml_loc"],
            f"pt{pretrain_id//100:02d}s",
            f"pt{pretrain_id:04d}.yaml"
        )
        self._prepare_fixed_params(pt_config_file, device)

        self._prepare_save_loc(pretrain_id)
        self._save_config()
        self._select_device(device, use_cpu)

        # For debugging.
        if debug:
            self.fixed_params.epochs = 2
            self.fixed_params.eval_every = 1
        self.debug = debug

    def _prepare_fixed_params(self, pt_config_file, device):
        """
        Args:
            fixed_params (Namespace): fixed parameters

        Returns:
            fixed_params (Namespace): updated fixed parameters
        """
        fixed_params, _, _ =\
            self._load_train_params(pt_config_file)
        fixed_params = self._update_fixed_params(fixed_params)
        fixed_params = self._insert_freq_to_fixed_param(fixed_params)
        fixed_params = Namespace(**fixed_params)

        # Add some fixed params.
        fixed_params.seed = config["experiment"]["seed"]["pretrain"]
        fixed_params.host = socket.gethostname()
        fixed_params.device = device
        self.fixed_params = fixed_params

    def _prepare_save_loc(self, pretrain_id: int):
        """
        Args:

        Returns:
            None
        """
        self.save_loc = os.path.join(
            config["experiment"]["path"]["save_root"],
            f"pt{pretrain_id:04d}"[:-2]+"s",
            f"pt{pretrain_id:04d}",
            get_timestamp()
        )
        os.makedirs(self.save_loc, exist_ok=True)

    def main(self):
        """
        Args:
            None
        Returns:
            None
        """
        pretrained_weight_dir = self._get_pt_dir()
        _, save_dir = run_train(
            args=self.fixed_params, 
            save_root=self.save_loc,
            run_pretraining=True,
            weight_file_dir=pretrained_weight_dir
        )
        print(save_dir)

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        '--pt', 
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

    executer = PretrainManager(
        int(args.pt), 
        args.device,
        debug=args.debug
    )
    executer.main()
