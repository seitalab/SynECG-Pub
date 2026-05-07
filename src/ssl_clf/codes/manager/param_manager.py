import socket
from argparse import Namespace

import yaml
import pandas as pd

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)

def update_dict_by_key(dict_x, dict_y):
    for key, value in dict_y.items():
        if key not in dict_x:
            dict_x[key] = value
    return dict_x

class ParamManager:

    def __init__(
        self, 
        yaml_file: str, 
        device: str, 
        is_pretrain: bool=False
    ):
        """
        Args:

        Returns:

        """

        self.is_pretrain = is_pretrain
        self._load_from_expyaml(yaml_file)

        update_dict = {
            "device": device,
            "host": socket.gethostname(),
        }
        self.update_params(update_dict)
        self._update_fixed_params()

    def update_params(self, update_dict, skip_if_exist: bool=False):
        """
        Args:
            update_dict (Dict):
            skip_if_exist (bool): If True, skip updating if key already exists.
        Returns:
            None
        """
        if skip_if_exist:
            if update_dict is not None:
                self.fixed_params = update_dict_by_key(self.fixed_params, update_dict)
        else:
            self.fixed_params.update(update_dict)

    def _update_by_config(self, key=None):

        key = "base" if key is None else key
        info_dict = config["ssl"]["eval_pt_model"]["params"][key]
                
        self.update_params(info_dict, skip_if_exist=True)

    def _update_fixed_params(self):
        # Add shared parameters from config.
        self._update_by_config()
        self._update_by_config(self.fixed_params["param_key"])

        # str -> float
        self._str_to_number("learning_rate", to_int=False)
        if self.is_pretrain:
            self._str_to_number("total_samples")
            self._str_to_number("eval_every")
            self._str_to_number("save_model_every")
            self._str_to_number("dump_every")

    def _str_to_number(self, key: str, to_int: bool=True):
        """
        Args:
            str_num (str): `XX*1eY`
        """
        if key not in self.fixed_params:
            return

        str_num = self.fixed_params[key].split("*")
        number = float(str_num[0]) * float(str_num[1])
        if to_int:
            number = int(number)
        self.fixed_params[key] = number

    def update_by_search_result(
        self, 
        score_sheet_path: str,
        metric_key: str="loss"
    ):
        """
        Args:
            score_sheet_path (str):
        Returns:
            None
        """
        # Open result sheet.
        df = pd.read_csv(score_sheet_path, index_col=0)

        if self.search_mode == "gs":
            # fetch metric col. and get best row index.
            result_col = df[metric_key]
            if metric_key == "loss":
                get_min = True
            else:
                get_min = False
        elif self.search_mode == "hps":
            result_col = df["value"]
            get_min = True
        else:
            raise NotImplementedError
        best_row_idx = result_col.idxmin() if get_min else result_col.idxmax()
        best_row = df.loc[best_row_idx]

        # Fetch target params from best row.
        update_dict = {}
        if self.search_params is not None:
            keys = self.search_params.keys()
        else:
            keys = [
                key.replace("params_", "")
                for key in best_row.keys() 
                if key.startswith("params_")
            ]
        for key in keys:
            if self.search_mode == "hps":
                result_key = f"params_{key}"
            update_dict[key] = best_row[result_key]

        # Update fixed params.
        self.update_params(update_dict)

    def _load_from_expyaml(self, yaml_file: str):
        """

        Args:
            config_file (str): _description_
        Returns:
            fix_params (Dict): 
            hps_mode (bool): True if hps False if grid search.
            search_params (Dict): hps_params or gs_params.
        """ 
        with open(yaml_file) as f:
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

        self.search_mode = search_mode
        self.search_params = search_params
        self.fixed_params = fixed_params

    def get_params(self, target: str="fixed_params"):
        if target == "fixed_params":
            return Namespace(**self.fixed_params)
        elif target == "search_params":
            return Namespace(**self.search_params)
        else:
            raise NotImplementedError
