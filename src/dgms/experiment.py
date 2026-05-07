import os
from argparse import Namespace

import yaml

from train import run_train
from codes.utils.utils import (
    get_timestamp, 
    ParameterManager,
)

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)

class ExperimentManagerBase:

    def _select_device(self, device: str, use_cpu: bool):
        """
        Args:

        Returns:
            None
        """
        if use_cpu:
            device = "cpu"
        self.device = device

    def _save_config(self):
        """

        Args:
            config_file (str): _description_
        Returns:
            None
        """
        # Convert `self.fixed_params` to dict.
        fixed_params = vars(self.fixed_params)

        # Save dict to yaml.
        save_loc = os.path.join(self.save_loc, "exp_config.yaml")
        with open(save_loc, "w") as f:
            yaml.dump(fixed_params, f)
        
    def _str_to_number(self, fixed_params, key, to_int: bool=True):
        """
        Args:
            str_num (str): `XX*1eY`
        """
        if key not in fixed_params:
            return fixed_params

        if type(fixed_params[key]) != str:
            return fixed_params

        str_num = fixed_params[key].split("*")
        number = float(str_num[0]) * float(str_num[1])

        if to_int:
            number = int(number)
        fixed_params[key] = number
        return fixed_params

    def _load_train_params(self, config_file: str):
        """

        Args:
            config_file (str): _description_
        Returns:
            fix_params (Dict): 
            hps_mode (bool): True if hps False if grid search.
            search_params (Dict): hps_params or gs_params.
        """ 
        with open(config_file) as f:
            params = yaml.safe_load(f)

        fixed_params = {}
        for key, value in params.items():
            if type(value) != dict:
                continue

            if value["param_type"] == "fixed":
                fixed_params[key] = value["param_val"]
            else:
                raise NotImplementedError

        return fixed_params

    def _insert_freq_to_fixed_param(self, fixed_params):
        
        dataset = fixed_params["dataset"]

        if dataset == "PTBXL-ALL":
            fixed_params["src_freq"] = 500
        else:
            raise ValueError(f"Dataset {dataset} not supported.")
        
        downsample = fixed_params["src_freq"] // fixed_params["target_freq"]
        fixed_params["downsample"] = downsample

        return fixed_params

    def _update_params(self, params, update_dict):
        if update_dict is not None:
            params.update(update_dict)
        return params

    def __update_by_config(self, fixed_params, dgm, key, arch=None):

        current = config["generatives"]
        if dgm is None:
            dgm = "common"
        
        if arch is not None:
            keys = [dgm, arch, key]
        else:
            keys = [dgm, key]
        
        to_update = True
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                to_update = False
                break
        fixed_params = self._update_params(fixed_params, current) if to_update else fixed_params
        return fixed_params

    def _update_fixed_params(self, fixed_params):

        key = fixed_params["exp_setting_key"] # eg. `prelim00`
        ssl = fixed_params["dgm"] # eg. `vae`
        # arch = fixed_params["architecture"] # eg. `resnet18`

        # Add shared parameters from config.
        fixed_params = self.__update_by_config(fixed_params, None, "base")
        fixed_params = self.__update_by_config(fixed_params, None, key)
        fixed_params = self.__update_by_config(fixed_params, ssl, "base", "all_arch")
        fixed_params = self.__update_by_config(fixed_params, ssl, key, "all_arch")
        # fixed_params = self.__update_by_config(fixed_params, ssl, "base", arch)
        # fixed_params = self.__update_by_config(fixed_params, ssl, key, arch)

        # str -> float
        fixed_params = self._str_to_number(fixed_params, "total_samples")
        fixed_params = self._str_to_number(fixed_params, "eval_every")
        fixed_params = self._str_to_number(fixed_params, "learning_rate", to_int=False)
        fixed_params = self._str_to_number(fixed_params, "save_model_every")
        fixed_params = self._str_to_number(fixed_params, "dump_every")
        return fixed_params

class ExperimentManager(ExperimentManagerBase):

    def __init__(
        self, 
        exp_id: int, 
        device: str,
        use_cpu: bool=False,
        debug: bool=False,
    ):
        """
        Args:
            exp_id (int): _description_
            device (str): cuda device or cpu to use.
            use_cpu (bool, optional): _description_. Defaults to False.
            debug (bool, optional): _description_. Defaults to False.
        """
        exp_config_file = os.path.join(
            config["experiment"]["path"]["dgm_yaml_loc"],
            f"d{exp_id//100:02d}s",
            f"dgm{exp_id:04d}.yaml"
        )
        
        # Load parameters.
        self._prepare_fixed_params(exp_config_file, device)
        self._prepare_save_loc(exp_id)
        self._save_config()
        self._select_device(device, use_cpu)

        # For debugging.
        if debug:
            self.fixed_params.data_lim = 1000
            self.fixed_params.val_lim = 400
            self.fixed_params.total_samples = 1000
            self.fixed_params.save_model_every = 400
            self.fixed_params.eval_every = 200
            self.fixed_params.dump_every = 500
        self.debug = debug
    
    def _prepare_fixed_params(self, exp_config_file, device):

        fixed_params =\
            self._load_train_params(exp_config_file)
        fixed_params = self._update_fixed_params(fixed_params)
        fixed_params = self._insert_freq_to_fixed_param(fixed_params)
        fixed_params = Namespace(**fixed_params)

        fixed_params.device = device
        self.fixed_params = fixed_params

    def _prepare_save_loc(self, exp_id: int):
        """
        Args:

        Returns:
            None
        """
        self.save_loc = os.path.join(
            config["experiment"]["path"]["save_root_gen"],
            f"dgm{exp_id:04d}"[:-2]+"s",
            f"dgm{exp_id:04d}",
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
        param_manager = ParameterManager(self.fixed_params)
        param_manager.add_param("device", self.device)
        param_manager.add_param(
            "seed", config["experiment"]["seed"]["pretrain"])
        param_manager.add_param(
            "is_gan", self.fixed_params.dgm in config["generatives"]["gans"])

        # os.makedirs(self.save_loc, exist_ok=True)
        train_params = param_manager.get_parameter()

        # Run training and store result.
        _, trained_model_loc = run_train(
            train_params, 
            self.save_loc, 
        )
        print(f"Trained model saved at {trained_model_loc}")

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

    executer = ExperimentManager(
        int(args.exp), 
        args.device,
        debug=args.debug
    )
    executer.main()
