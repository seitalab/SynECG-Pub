import os
import gc
import pickle
from argparse import Namespace
from typing import Dict

import yaml
import torch
import optuna
from optuna.pruners import PatientPruner, MedianPruner

from codes.run_train import run_train
from codes.manager.result_manager import TemporalResultSaver

torch.backends.cudnn.deterministic = True

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)

def sample_params(
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


# Used for hyperparameter search with TPE.
class Objective:

    def __init__(
        self, 
        base_args: Namespace, 
        search_space: Dict, 
        save_root: str,
        train_info: str=None,
    ) -> None:
        """
        Args:
            base_args (Namespace): Hyper parameters fixed during training.
            search_space (Dict): Dictionary of search space for hyper paramter optimization.
                {"arg_name": [range_type, low, high]}
            save_root (str): 
            train_info (str):
        Returns:
            None
        """
        self.args = base_args
        self.search_space = search_space
        self.train_info = train_info
        self.save_root = save_root

        self._seen_params = set()

        # Check if namespace has `debug` attribute.
        if hasattr(base_args, "debug"):
            self.debug = base_args.debug
        else:
            self.debug = False

    def _prep_params_tracker(self, params: Namespace) -> str:
        """
        Args:
            params (Namespace):
        Returns:
            params_tracker (str):
        """
        sorted_keys = sorted(self.search_space.keys())
        param_strs = [f"{k}={getattr(params, k)}" for k in sorted_keys]
        return "|".join(param_strs)

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """
        Args:
            trial (optuna.trial.Trial): 
        Returns:
            best_loss (float):
        """
        params = sample_params(
            self.args, 
            self.search_space,
            trial
        )

        # Avoid running same setting multiple times.
        params_id = self._prep_params_tracker(params)
        if params_id in self._seen_params:
            raise optuna.TrialPruned()
        self._seen_params.add(params_id)

        # Avoid errors from transformer.
        if ("heads" in vars(params) and "emb_dim" in vars(params)):
            if params.heads > params.emb_dim:
                raise optuna.TrialPruned()
        if "qkv_dim" in vars(params):
            if params.heads > params.qkv_dim:
                raise optuna.TrialPruned()
        
        # Run train.
        if self.debug:
            best_loss = self._run_train(params, trial)
        else:
            try:
                best_loss = self._run_train(params, trial)        
            except Exception as e:
                print(str(e))
                best_loss = 1e5

        gc.collect()
        torch.cuda.empty_cache()
        return best_loss
    
    def _run_train(self, params, trial):
        best_result_dict, _ = run_train(
            params, self.save_root, trial=trial)
        best_loss = best_result_dict["loss"]
        return best_loss

def run_hps(
    args: Namespace,
    save_root: str,
    search_space: Dict,
) -> str:
    """
    Args:
        args (Namespace):
        save_root (str): 
        search_space (Dict): Hyperparameter search space.
    Returns:
        csv_name (str):
    """
    assert ("num_trials" in args and "max_time" in args)

    # Prepare result storing directories
    save_loc = os.path.join(save_root, "hps/runs")
    os.makedirs(save_loc, exist_ok=True)

    study = optuna.create_study(
        sampler = optuna.samplers.TPESampler(seed=args.seed),
        direction="minimize",
        pruner=PatientPruner(
            MedianPruner(), 
            patience=5
        ),
    )    
    # catch: Continue when `RuntimeError: CUDA out of memory`.
    # callbacks: Save temporal result
    tmp_saver = TemporalResultSaver(save_loc)
    study.optimize(
        Objective(args, search_space, save_loc), 
        n_trials=args.num_trials, 
        timeout=args.max_time, 
        catch=(RuntimeError,), 
        callbacks=[tmp_saver.save_temporal_result],
        n_jobs=1
    )
    
    # Save study result as csv.
    csv_name = os.path.join(save_loc, "../result.csv")
    df_hps = study.trials_dataframe()
    df_hps.to_csv(csv_name)

    # Save study as pickle.
    pkl_name = os.path.join(save_loc, "../result.pkl")
    with open(pkl_name, "wb") as fp:
        pickle.dump(study, fp)

    return csv_name

if __name__ == "__main__":
    pass