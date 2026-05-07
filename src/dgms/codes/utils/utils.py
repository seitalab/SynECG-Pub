import socket
from datetime import datetime
from typing import Dict, List, Union
from argparse import Namespace

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)

colors = ["k", "c", "b", "m"]

f01 = 20
f02 = 16

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    return specificity

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

def debug_draw_samples(ecgs, save_loc):
    import os
    os.makedirs(save_loc, exist_ok=True)
    ecgs = ecgs.cpu().detach().numpy()
    for i, ecg in enumerate(ecgs):
        make_ecg_plot(ecg[0], 10, 500, f"{save_loc}/{i+1:04d}.png")
        if i == 4:
            break
    

def make_ecg_plot(
    ecg: np.ndarray, 
    ecg_duration: float,
    frequency: int,
    savename: str=None, 
    n_ecg: int=1
) -> None:
    """
    Args:
        ecg (np.ndarray): Array of size (data length,).
        ecg_duration (float): 
        frequency (int): 
        savename (str): Filename for saving.
    Returns:
        None
    """
    cm = 1 / 2.54

    # Define figure width.
    # 2.5 [cm / sec]: definition by ecg plot.
    # 2.5 [cm / sec] * `per_fig_length` [sec] 
    width = ecg_duration * 2.5

    x_scale_buffer = 10

    if n_ecg > 1:
        ecg_len = len(ecg[0])
        ecg_abs = [
            np.abs(ecg[i]).max() for i in range(n_ecg)
        ]
        ecg_abs = max(ecg_abs)
    else:
        ecg_len = len(ecg)
        ecg_abs = np.abs(ecg).max()

    # Define figure height.
    height = 4 * 2
    if ecg_abs > 2:
        y_min = -4
        y_max = 4
        major_y_ticks = np.arange(y_min, y_max+2, 2)
        minor_y_ticks = np.arange(y_min, y_max+0.2, 0.2)
        # height *= 2
    else:
        y_min = -2
        y_max = 2
        major_y_ticks = np.arange(y_min, y_max+1)
        minor_y_ticks = np.arange(y_min, y_max+0.1, 0.1)
    y_scale_buffer = 0.1

    fig_height = height * cm
    fig_width = width * cm

    fig = plt.figure(
        figsize=(fig_width, fig_height)
    )
    ax = fig.add_subplot(1, 1, 1)


    major_x_ticks = np.arange(ecg_len+1)[::frequency]
    minor_x_ticks = np.arange(ecg_len+1)[::frequency//10]

    ax.set_xticks(major_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    ax.set_yticks(major_y_ticks)
    ax.set_yticks(minor_y_ticks, minor=True)

    if n_ecg > 1:
        for ecg_idx in range(n_ecg):
            ax.plot(ecg[ecg_idx], color=colors[ecg_idx])
    else:
        ax.plot(ecg, color="k")

    x_labels = [
        str(t//frequency) if t % frequency == 0 else "" 
        for t in major_x_ticks
    ]
    ax.set_xticklabels(x_labels, fontsize=f02)
    ax.set_xlabel("[sec]", fontsize=f01)
    ax.set_xlim(-1 * x_scale_buffer, ecg_len + x_scale_buffer)

    # ax.set_ylabel("[mV]")
    ax.set_ylabel("scaled amplitude", fontsize=f01)
    ax.set_ylim(y_min - y_scale_buffer, y_max + y_scale_buffer)

    plt.grid(
        visible=True, 
        axis="both", 
        which="major", 
        color="y", 
        linestyle="-", 
        linewidth=1
    )
    plt.grid(
        visible=True, 
        axis="both", 
        which="minor", 
        color="y", 
        linestyle="dotted", 
        alpha=0.3, 
        linewidth=1
    )

    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    
    if savename is not None:
        plt.savefig(savename, bbox_inches="tight")
        plt.close()
    else:
        return fig

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


