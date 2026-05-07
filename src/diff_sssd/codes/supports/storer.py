import os
import json
import math
import pickle
from typing import Dict

import torch
import torch.nn as nn

class Storer:

    def __init__(
        self, 
        save_dir: str, 
        store_interim_model: bool=False,
        enabled: bool=True,
    ):
        """
        Args:
            save_dir (str): Path to save dir
        Returns:
            None
        """
        self.enabled = enabled
        if self.enabled:
            os.makedirs(save_dir, exist_ok=True)
            if store_interim_model:
                os.makedirs(save_dir+"/interims", exist_ok=True)
        self.save_dir = save_dir

        self.trains = {"loss": {}, "score": {}}
        self.evals = {"loss": {}, "score": {}}

    def _state_dict(self, model: nn.Module):
        module = model.module if hasattr(model, "module") else model
        return module.state_dict()

    def save_params(self, params) -> None:
        """
        Save parameters.
        Args:
            params
        Returns:
            None
        """
        if not self.enabled:
            return
        savename = self.save_dir + "/params.pkl"
        with open(savename, "wb") as fp:
            pickle.dump(params, fp)

    def save_model(self, model: nn.Module, score: float) -> None:
        """
        Save current model (overwrite existing model).
        Args:
            model (nn.Module):
            score (float):
        Returns:
            None
        """
        if not self.enabled:
            return
        savename = self.save_dir + "/net.pth"
        torch.save(self._state_dict(model), savename)

        with open(self.save_dir + "/best_score.txt", "w") as f:
            f.write(f"{score:.5f}")

    def store_epoch_result(
        self, 
        epoch: int, 
        epoch_result_dict: Dict, 
        is_eval: bool = False
    ) -> None:
        """
        Args:
            epoch (int):
            score (float):
        Returns:
            None
        """
        if not self.enabled:
            return

        if is_eval:
            self.evals["loss"][epoch] = epoch_result_dict["loss"]
            self.evals["score"][epoch] = epoch_result_dict["score"]
        else:
            self.trains["loss"][epoch] = epoch_result_dict["loss"]
            self.trains["score"][epoch] = epoch_result_dict["score"]

    def store_logs(self):
        """
        Args:
            None
        Returns:
            None
        """
        if not self.enabled:
            return
        with open(self.save_dir + "/train_scores.json", "w") as ft:
            json.dump(self.trains, ft, indent=4)

        with open(self.save_dir + "/eval_scores.json", "w") as fe:
            json.dump(self.evals, fe, indent=4)

    def save_model_interim(self, model, n_sample, denom=1e6, n_iter=None):
        """
        Args:

        Returns:
            None
        """
        if not self.enabled:
            return
        power = round(math.log(denom, 10), 3)
        n_sample_d = n_sample / denom
        info = f"{int(n_sample_d):06d}E{power:.2f}"

        savename = self.save_dir + f"/interims/net_{info}.pth"
        if not os.path.exists(os.path.dirname(savename)):
            os.makedirs(os.path.dirname(savename), exist_ok=True)
        state_dict = self._state_dict(model)
        torch.save(state_dict, savename)

        # Backward compatibility with legacy SSSD checkpoint naming:
        # save raw state_dict as "<iter>.pkl" at the run root.
        if n_iter is not None:
            legacy_savename = self.save_dir + f"/{int(n_iter)}.pkl"
            torch.save(state_dict, legacy_savename)
