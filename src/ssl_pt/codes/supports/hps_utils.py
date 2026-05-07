import os
from argparse import Namespace
from typing import Dict

import optuna

def prepare_params(
    args: Namespace, 
    search_space: Dict, 
    trial: optuna.trial.Trial
) -> Namespace:
    """
    Concatenate `base_args` and parameters sampled from `search_space`,
    return as single Namespace.
    Args:
        trial (optuna.trial.Trial):
    Returns:
        params (Namespace):
    """
    params = vars(args) # Namespace -> Dict
    for variable, sample_info in search_space.items():
        if sample_info[0] == "int":
            _param = trial.suggest_int(
                variable, 
                sample_info[1], 
                sample_info[2]
            )
        elif sample_info[0] == "uniform":
            _param = trial.suggest_float(
                variable, 
                sample_info[1], 
                sample_info[2]
            )
        elif sample_info[0] == "log_uniform":
            _param = trial.suggest_float(
                variable, 
                sample_info[1], 
                sample_info[2],
                log=True
            )
        elif sample_info[0] == "discrete_uniform":
            _param = trial.suggest_int(
                variable, 
                sample_info[1], 
                sample_info[2], 
                step=sample_info[3]
            )
        elif sample_info[0] == "int_pow":
            _param = trial.suggest_int(
                variable, 
                sample_info[1], 
                sample_info[2]
            )
            _param = sample_info[3] ** _param
        elif sample_info[0] == "categorical":
            _param = trial.suggest_categorical(
                variable, 
                sample_info[1]
            )
        else:
            raise NotImplementedError
        params.update({variable: _param})

    # Overwrite `epochs`.
    if "hps_epochs" in params.keys():
        params["epochs"] = params["hps_epochs"]
    
    params = Namespace(**params) # Dict -> Namespace
    return params

class TemporalResultSaver:

    def __init__(self, save_loc: str) -> None:
        """
        Args:
            save_loc (str): 
        Returns:
            None
        """
        self.save_loc = save_loc

    def save_temporal_result(self, study, frozen_trial):
        """
        Arguments are required by optuna.
        Args:
            study: 
            frozen_trial: <REQUIRED BY OPTUNA>
        Returns:
            None
        """
        filename = "tmp_result_hps.csv"
        csv_name = os.path.join(
            self.save_loc, filename)
        df_hps = study.trials_dataframe()
        df_hps.to_csv(csv_name)

