import os
from typing import Optional
from argparse import Namespace
from itertools import product
from glob import glob

import yaml

from codes.run_train import run_train
from codes.supports.utils import (
    get_timestamp, 
    ParameterManager,
    ResultManager, 
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
        
    def _str_to_number(self, str_num: str, to_int: bool=True):
        """
        Args:
            str_num (str): `XX*1eY`
        """
        # if `str_num` is not string return as it is.
        if not isinstance(str_num, str):
            return str_num

        str_num = str_num.split("*")
        number = float(str_num[0]) * float(str_num[1])

        if to_int:
            number = int(number)
        return number

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

        fixed_params, hps_params, gs_params = {}, {}, {}
        for key, value in params.items():
            if type(value) != dict:
                continue

            if value["param_type"] == "fixed":
                fixed_params[key] = value["param_val"]
            elif value["param_type"] == "grid":
                assert type(value["param_val"]) == list
                gs_params[key] = value["param_val"] # List stored
            elif value["param_type"] == "hps":
                assert type(value["param_val"]) == list
                hps_params[key] = value["param_val"]
            else:
                raise NotImplementedError

        # hps_params and gs_params must not have value at same time.
        assert not (bool(hps_params) and bool(gs_params))
        if (bool(hps_params) and not bool(gs_params)):
            search_mode = "hps"
            search_params = hps_params
        elif (not bool(hps_params) and bool(gs_params)):
            search_mode = "gs"
            search_params = gs_params
        elif (not bool(hps_params) and not bool(gs_params)):
            search_mode = None
            search_params = None
        else:
            raise        

        return fixed_params, search_mode, search_params

    def _insert_freq_to_fixed_param(self, fixed_params):
        
        dataset = fixed_params["dataset"]

        fixed_params["src_freq"] = config["settings"]["common"]["target_freq"]
        
        downsample = fixed_params["src_freq"] // fixed_params["target_freq"]
        fixed_params["downsample"] = downsample

        return fixed_params

    def _update_params(self, params, update_dict):
        if update_dict is not None:
            params.update(update_dict)
        return params

    def __update_by_config(self, fixed_params, ssl, key, arch=None):

        current = config["ssl"][self.exp_mode]
        if ssl is None:
            ssl = "common"
        
        if arch is not None:
            keys = [ssl, arch, key]
        else:
            keys = [ssl, key]
        
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
        ssl = fixed_params["ssl"] # eg. `mae`
        arch = fixed_params["architecture"] # eg. `resnet18`

        # Add shared parameters from config.
        fixed_params = self.__update_by_config(fixed_params, None, "base")
        fixed_params = self.__update_by_config(fixed_params, None, key)
        fixed_params = self.__update_by_config(fixed_params, ssl, "base", "all_arch")
        fixed_params = self.__update_by_config(fixed_params, ssl, key, "all_arch")
        fixed_params = self.__update_by_config(fixed_params, ssl, "base", arch)
        fixed_params = self.__update_by_config(fixed_params, ssl, key, arch)

        # str -> float
        if "total_samples" in fixed_params:
            fixed_params["total_samples"] = self._str_to_number(
                fixed_params["total_samples"])
        if "eval_every" in fixed_params:
            fixed_params["eval_every"] = self._str_to_number(
                fixed_params["eval_every"])
        if "learning_rate" in fixed_params:
            fixed_params["learning_rate"] = self._str_to_number(
                fixed_params["learning_rate"], to_int=False)
        if "save_model_every" in fixed_params:
            fixed_params["save_model_every"] = self._str_to_number(
                fixed_params["save_model_every"])
        
        return fixed_params

    def _get_pt_dir(self):
        key = self.fixed_params.pretrained_weight_key
        if key is None:
            return None
        
        key_to_dir_file = "./resources/settings.yaml"
        with open(key_to_dir_file) as f:
            key_to_dir = yaml.safe_load(f)
        weight_dir = key_to_dir["pt_model"]\
            [self.fixed_params.ssl]\
            [self.fixed_params.architecture]\
            [self.fixed_params.pretrained_weight_key]
        return weight_dir

    def _merge_from_pretrain_setting(self, pretrained_weight_dir):
        if pretrained_weight_dir is None:
            return None

        yaml_file = glob(pretrained_weight_dir + "/../exp_config.yaml")
        assert len(yaml_file) == 1

        with open(yaml_file[0]) as f:
            mae_cfg = yaml.safe_load(f)

        fixed_params = vars(self.fixed_params)
        for target_param in fixed_params["reuse_params"]:
            fixed_params[target_param] =\
                mae_cfg[target_param]

        del fixed_params["reuse_params"]
        self.fix_params = Namespace(**fixed_params)

class ExperimentManager(ExperimentManagerBase):

    def __init__(
        self, 
        exp_id: int, 
        device: str,
        use_cpu: bool=False,
        debug: bool=False,
        pt_eval: bool=True,
    ):
        """
        Args:
            exp_id (int): _description_
            device (str): cuda device or cpu to use.
            use_cpu (bool, optional): _description_. Defaults to False.
            debug (bool, optional): _description_. Defaults to False.
        """
        if pt_eval:
            exp_config_file = os.path.join(
                config["experiment"]["path"]["pt_eval_yaml_loc"],
                f"pt_eval{exp_id//100:02d}s",
                f"pt_eval{exp_id:04d}.yaml"
            )
            self.exp_mode = "pt_eval"
        else:
            exp_config_file = os.path.join(
                config["experiment"]["path"]["yaml_loc"],
                f"exp{exp_id//100:02d}s",
                f"exp{exp_id:04d}.yaml"
            )
            self.exp_mode = "clf"

        # Load parameters.
        self._prepare_fixed_params(exp_config_file, device)

        self._prepare_save_loc(exp_id, pt_eval)
        self._save_config()
        self._select_device(device, use_cpu)

        # For debugging.
        if debug:
            self.fixed_params.epochs = 2
            self.fixed_params.eval_every = 1
        self.debug = debug
    
    def _prepare_fixed_params(self, exp_config_file, device):

        fixed_params, self.search_mode, search_params =\
            self._load_train_params(exp_config_file)
        fixed_params = self._update_fixed_params(fixed_params)
        fixed_params = self._insert_freq_to_fixed_param(fixed_params)
        fixed_params = Namespace(**fixed_params)
        self.search_params = \
            Namespace(**search_params) if search_params is not None else None
        
        fixed_params.device = device
        self.fixed_params = fixed_params

    def _prepare_save_loc(self, exp_id: int, pt_eval: bool):
        """
        Args:

        Returns:
            None
        """
        if pt_eval:
            self.save_loc = os.path.join(
                config["path"]["processed_data"],
                config["experiment"]["path"]["save_root"],
                f"pt-eval{exp_id:04d}"[:-2]+"s",
                f"pt_eval{exp_id:04d}",
                get_timestamp()
            )
        else:
            self.save_loc = os.path.join(
                config["path"]["processed_data"],
                config["experiment"]["path"]["save_root"],
                f"exp{exp_id:04d}"[:-2]+"s",
                f"exp{exp_id:04d}",
                get_timestamp()
            )
        os.makedirs(self.save_loc, exist_ok=True)

    def run_gs_experiment(self) -> str:
        """
        Args:
            None

        Returns:
            csv_path (str): _description_
        """
        # Prepare parameters.
        param_manager = ParameterManager(self.fixed_params)
        param_manager.add_param(
            "seed", self.fixed_params.search_seed)
        param_manager.add_param("device", self.device)

        # Prepare search space.
        search_space_dict = vars(self.search_params)
        search_space = list(
            product(*search_space_dict.values())
        )
        search_keys = list(search_space_dict.keys())

        # Prepare result storer.
        columns = search_keys + config["experiment"]["result_cols"] + ["save_loc"]

        savename = os.path.join(self.save_loc, "result_table_gs.csv")
        result_manager = ResultManager(savename=savename, columns=columns)

        # Execute grid search.
        for param_comb in search_space:
            
            # Prepare training param.
            for i, param in enumerate(param_comb):
                param_manager.add_param(search_keys[i], param)
            train_params = param_manager.get_parameter()

            save_loc = os.path.join(self.save_loc, "gs")
            # Run training and store result.
            best_val_result, result_save_dir = run_train(
                train_params, 
                save_loc,
                finetune_target=train_params.pretrained_model
            )

            # Form result row.
            result_row = list(param_comb)
            for key in columns[len(search_keys):]:
                if key not in best_val_result:
                    continue
                result_row.append(best_val_result[key])
            result_row.append(result_save_dir)

            # Store result row.
            result_manager.add_result(result_row)
            result_manager.save_result(is_temporal=True)
        
        result_manager.save_result()
        return result_manager.savename

    def run_hps_experiment(self) -> str:
        """
        Args:
            None
        Returns:
            savename (str): _description_
        """
        # Prepare parameters.
        param_manager = ParameterManager(self.fixed_params)
        param_manager.add_param("seed", config["experiment"]["seed"]["hps"])
        param_manager.add_param("device", self.device)

        # Execute hyper parameter search.
        train_params = param_manager.get_parameter()
        csv_name = run_hps(
            train_params, 
            self.save_loc, 
            vars(self.search_params)
        )

        # Copy hyperparameter result file.
        savename = os.path.join(
            self.save_loc, f"ResultTableHPS.csv")
        os.system(f"cp {csv_name} {savename}")

        return savename

    def run_multiseed_experiment(
        self, 
        score_sheet: Optional[str], 
        single_run: bool,
    ):
        """_summary_

        Args:
            score_sheet (str): _description_
        """        
        param_manager = ParameterManager(self.fixed_params)
        param_manager.add_param("device", self.device)

        if score_sheet is not None:
            param_manager.update_by_search_result(
                score_sheet, 
                vars(self.search_params), 
                is_hps=self.search_mode == "hps"
            )
        
        # Prepare result storer.
        columns = ["seed", "dataset"] + config["experiment"]["result_cols"]        
        savename = os.path.join(self.save_loc, "ResultTableMultiSeed.csv")
        result_manager = ResultManager(savename=savename, columns=columns)

        for s_idx, seed in enumerate(config["experiment"]["seed"]["multirun"]):
            
            param_manager.add_param("seed", seed)
            save_loc_train = os.path.join(
                self.save_loc, "multirun", "train", f"seed{seed:04d}")
            os.makedirs(save_loc_train, exist_ok=True)
            train_params = param_manager.get_parameter()

            # Run training and store result.
            _, trained_model_loc = run_train(
                train_params, 
                save_loc_train, 
                finetune_target=train_params.pretrained_model
            )

            # Eval.
            save_loc_eval = os.path.join(
                self.save_loc, "multirun", "eval", f"seed{seed:04d}")
            os.makedirs(save_loc_eval, exist_ok=True)

            val_result, test_result = run_eval(
                eval_target=trained_model_loc, 
                device=self.device,
                dump_loc=save_loc_eval, 
                multiseed_run=True
            )
            result_row = self._form_result_row(
                seed, "val", columns, val_result)
            result_manager.add_result(result_row)
            # result_row = self._form_result_row(
            #     seed, "test(internal)", columns, test_i_result)
            # result_manager.add_result(result_row)
            result_row = self._form_result_row(
                seed, "test(external)", columns, test_e_result)
            result_manager.add_result(result_row)
            
            result_manager.save_result(is_temporal=True)

            if single_run:
                break

        result_manager.save_result()
        return result_manager.savename

    def _form_result_row(self, seed, dataset, columns, result_dict):
        """
        Args:

        Returns:

        """

        result_row = [seed, dataset]
        for key in columns[2:]:
            if key not in result_dict:
                result_row.append(None)
            else:
                result_row.append(result_dict[key])
        return result_row

    def main(self, single_run=False, hps_result=None):
        """
        Args:
            None
        Returns:
            None
        """
        # Overwrite.
        if self.fixed_params.hps_result is not None:
            assert hps_result is None
            hps_result = self.fixed_params.hps_result

        # Search.
        if hps_result is None:
            if self.search_mode == "hps": 
                csv_path = self.run_hps_experiment()
            elif self.search_mode == "gs":
                csv_path = self.run_gs_experiment()
            else:
                csv_path = None
        else:
            csv_path = hps_result
        
        # Multi seed eval.
        result_file = self.run_multiseed_experiment(
            csv_path, single_run)
        
        # End

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
    parser.add_argument(
        '--multirun', 
        action="store_true"
    )    
    args = parser.parse_args()

    print(args)

    executer = ExperimentManager(
        int(args.exp), 
        args.device,
        debug=args.debug
    )
    executer.main(not args.multirun)