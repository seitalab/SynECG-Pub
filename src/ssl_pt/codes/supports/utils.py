from datetime import datetime
from typing import List, Union, Dict, Tuple

import yaml
import numpy as np
import pandas as pd
# from slack_sdk import WebClient

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)
slack_config = config["slack"]

def get_timestamp() -> str:
    """
    Get timestamp in `yymmdd-hhmmss` format.
    Args:
        None
    Returns:
        timestamp (str): Time stamp in string.
    """
    timestamp = datetime.now()
    timestamp = timestamp.strftime('%Y%m%d-%H%M%S')[2:]
    return timestamp

def calc_class_weight(
    labels: List
) -> np.ndarray:
    """
    Calculate class weight for multilabel task.
    Args:
        labels (np.ndarray): Label data array of shape [num_sample, num_classes]
    Returns:
        class_weight (np.ndarray): Array of shape [num_classes].
    """
    labels = np.array(labels)[:, np.newaxis]
    num_samples = labels.shape[0]

    positive_per_class = labels.sum(axis=0)
    negative_per_class = num_samples - positive_per_class

    class_weight = negative_per_class / positive_per_class

    return class_weight

# class SlackReporter:

#     def __init__(self):
#         """
#         Args:
#             None
#         Returns:
#             None
#         """        
#         self.client = WebClient(
#             token=slack_config["token"]
#         )
#         self.channel_id = slack_config["channel_id"]

#     def report(
#         self, 
#         message: str,
#         parent_message: str, 
#     ) -> None:
#         """
#         Args:
#             message (str): 
#             parent_message (str): 
#         Returns:
#             None
#         """
#         history = self.client.conversations_history(
#             channel=self.channel_id
#         )
#         posts = history["messages"][:slack_config["max_past"]]

#         for post in posts:
#             if post["text"] != parent_message:
#                 continue
#             self.client.chat_postMessage(
#                 channel=self.channel_id,
#                 thread_ts=post["ts"],
#                 text=message
#             )
#             break
    
#     def post(self, message: str):
#         """
#         Args:
#             message (str): 
#         Returns:
#             None
#         """
#         self.client.chat_postMessage(
#             text=message, 
#             channel=self.channel_id,
#         )

class ResultManager:

    def __init__(self, savename: str, columns: List):
        self.savename = savename
        self.columns = columns
        self.results = []

    def add_result(self, row: List):
        """
        Add one row to results.
        
        Args:
            row (List): _description_
        Returns: 
            None
        """
        self.results.append(row)

    def get_result_df(self) -> pd.DataFrame:
        """
        Args:
            None
        Returns:
            df_result: 
        """
        df_result = pd.DataFrame(
            self.results, columns=self.columns)
        return df_result
    
    def save_result(self, is_temporal: bool=False):
        """

        Args:
            is_temporal (bool, optional): _description_. Defaults to False.
        """
        df_result = pd.DataFrame(
            self.results, columns=self.columns)
        
        savename = self.savename
        if is_temporal:
            savename = savename.replace(".csv", "_tmp.csv")
        df_result.to_csv(savename)


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
            target_metric = "loss"
            self.update_by_gs_result(
                result_file, searched_params, target_metric)

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
            param_val = best_setting[f"params_{param}"].item()
            if searched_params[param][0] == "int_pow":
                param_val = 2 ** param_val
            if searched_params[param][0] == "discrete_uniform":
                param_val = int(param_val)
            self.add_param(param, param_val)
        print(self.params)

    def update_by_gs_result(
        self, 
        gs_result_file: str, 
        searched_params: Dict,
        target_metric: str
    ) -> None:
        """

        Args:
            gs_result_file (str): _description_
            searched_params (Dict): _description_
        """
        # Open result csv.
        df_result = pd.read_csv(gs_result_file, index_col=0)

        # Find best value setting.
        best_row = np.argmin(df_result.loc[:, target_metric].values)
        best_setting = df_result.iloc[best_row]

        for param in searched_params.keys():
            try:
                param_val = best_setting[param].item()
            except:
                param_val = best_setting[param]
            self.add_param(param, param_val)
        print(self.params)

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
