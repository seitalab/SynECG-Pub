import socket
from typing import Dict, Union
from argparse import Namespace

import numpy as np
import pandas as pd

class ParameterManager:

    def __init__(self, base_params=None):

        if base_params is None:
            self.params = Namespace()
        else:
            self.params = base_params
        self.hps_params = Namespace()

        self.params.host = socket.gethostname()

    def add_param(
        self,
        param_key: str, 
        param_value: Union[int, float, str], 
    ):
        """
        Args:
            param_value (Union[int, float, str]): 
            param_key (str): 
        Returns:

        """
        params = vars(self.params)    
        params.update(
            {
                param_key: param_value
            }
        )
        self.params = Namespace(**params)

    def update_by_search_result(
        self, 
        result_file: str, 
        searched_params: Dict, 
        is_hps: bool
    ) -> None:
        """

        Args:
            result_file (str): _description_
            searched_params (Dict): _description_
            is_hps (bool): _description_
        """

        if is_hps:
            self.update_by_hps_result(result_file, searched_params)
        else:
            self.update_by_gs_result(result_file, searched_params)

    def update_by_hps_result(
        self, 
        hps_result_file: str, 
        searched_params: Dict
    ) -> None:
        """
        Args:
            hps_result_file (str): Result dirname for hyperparameter search result (assuming result saved as `result.csv`).
            searched_params (Dict): 
        Returns:
            None
        """
        df_result = pd.read_csv(hps_result_file, index_col=0)

        # Remove Failed trial.
        target_row = df_result.loc[:, "value"].notna().values
        df_result = df_result[target_row]

        # Find best value setting.
        best_row = np.argmin(df_result.loc[:, "value"].values)
        best_setting = df_result.iloc[best_row]

        for param in searched_params.keys():
            assert f"params_{param}" in best_setting.keys()
            try:
                param_val = best_setting[f"params_{param}"].item()
            except:
                param_val = best_setting[f"params_{param}"]
            if searched_params[param][0] == "int_pow":
                param_val = 2 ** param_val
            if searched_params[param][0] == "discrete_uniform":
                param_val = int(param_val)
            self.add_param(param, param_val)
        print(self.params)
    
    # def update_by_gs_result(
    #     self, 
    #     gs_result_file: str, 
    #     searched_params: Dict
    # ) -> None:
    #     """

    #     Args:
    #         gs_result_file (str): _description_
    #         searched_params (Dict): _description_
    #     """
    #     # Open result csv.
    #     df_result = pd.read_csv(gs_result_file, index_col=0)

    #     # Find best value setting.
    #     target_metric = config["fixed_setting"]["gs_monitor_metric"]
    #     best_row = np.argmin(df_result.loc[:, target_metric].values)
    #     best_setting = df_result.iloc[best_row]

    #     for param in searched_params.keys():
    #         param_val = best_setting[param].item()
    #         self.add_param(param, param_val)
    #     print(self.params)

    def get_parameter(self):
        """
        Args:

        Returns:

        """
        return self.params

    def get_hps_parameter(self):
        """
        Args:

        Returns:

        """
        return self.hps_params