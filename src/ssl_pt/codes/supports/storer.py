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
        store_interim_model: bool=False
    ):
        """
        Args:
            save_dir (str): Path to save dir
        Returns:
            None
        """
        os.makedirs(save_dir, exist_ok=True)
        if store_interim_model:
            os.makedirs(save_dir+"/interims", exist_ok=True)
        self.save_dir = save_dir

        self.trains = {"loss": {}, "score": {}}
        self.evals = {"loss": {}, "score": {}}

    def save_params(self, params) -> None:
        """
        Save parameters.
        Args:
            params
        Returns:
            None
        """
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
        savename = self.save_dir + "/net.pth"
        torch.save(model.state_dict(), savename)

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

        with open(self.save_dir + "/train_scores.json", "w") as ft:
            json.dump(self.trains, ft, indent=4)

        with open(self.save_dir + "/eval_scores.json", "w") as fe:
            json.dump(self.evals, fe, indent=4)

    def save_model_interim(self, model, n_sample, denom=1e6):
        """
        Args:

        Returns:
            None
        """
        power = round(math.log(denom, 10), 3)
        n_sample_d = n_sample / denom
        info = f"{int(n_sample_d):06d}E{power:.2f}"

        savename = self.save_dir + f"/interims/net_{info}.pth"
        torch.save(model.state_dict(), savename)            
