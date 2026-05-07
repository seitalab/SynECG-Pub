import os
import json
import math
import pickle
from typing import Dict, List

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    f1_score, 
    roc_auc_score, 
    confusion_matrix,
    accuracy_score, 
    average_precision_score,
    recall_score,
    precision_score
)

from codes.utils.utils import sigmoid, specificity_score, make_ecg_plot

class Monitor:

    def __init__(self, target_keys: List=["total_loss"]) -> None:
        """
        Args:
            None
        Returns:
            None
        """
        self.num_data = 0
        self.total_losses = {key: 0 for key in target_keys}
        self.ytrue_record = None
        self.ypred_record = None

        self.inputs = None

    def _concat_array(self, record, new_data: np.array) -> np.ndarray:
        """
        Args:
            record ()
            new_data (np.ndarray):
        Returns:
            concat_data (np.ndarray):
        """
        if record is None:
            return new_data
        else:
            return np.concatenate([record, new_data])

    def store_loss(self, loss: float, target_key: str="total_loss") -> None:
        """
        Args:
            loss (float): Mini batch loss value.
        Returns:
            None
        """
        self.total_losses[target_key] += loss

    def store_num_data(self, num_data: int) -> None:
        """
        Args:
            num_data (int): Number of data in mini batch.
        Returns:
            None
        """
        self.num_data += num_data

    def store_result(self, y_trues: np.ndarray, y_preds: np.ndarray) -> None:
        """
        Args:
            y_trues (np.ndarray):
            y_preds (np.ndarray): Array with 0 - 1 values.
        Returns:
            None
        """
        y_trues = y_trues.cpu().detach().numpy()
        y_preds = y_preds.cpu().detach().numpy()

        self.ytrue_record = self._concat_array(self.ytrue_record, y_trues)
        self.ypred_record = self._concat_array(self.ypred_record, y_preds)
        assert(len(self.ytrue_record) == len(self.ypred_record))

    def store_input(self, input_x):

        input_x = input_x.cpu().detach().numpy()

        self.inputs = self._concat_array(self.inputs, input_x)

    def average_loss(self, target_key: str="total_loss") -> float:
        """
        Args:
            None
        Returns:
            average_loss (float):
        """
        return self.total_losses[target_key] / self.num_data

    def calc_f1(self, use_macro: bool=False) -> float:
        """
        Args:
            None
        Returns:
            score (float): F1 score.
        """
        y_preds = sigmoid(self.ypred_record)
        y_preds = (y_preds > 0.5).astype(int)
        if use_macro:
            score = f1_score(self.ytrue_record, y_preds, average='macro')
        else:
            score = f1_score(self.ytrue_record, y_preds, zero_division=0)
        return score

    def accuracy(self) -> float:
        """
        Args:
            None
        Returns:
            score (float):
        """            
        # y_preds = np.argmax(self.ypred_record, axis=1)
        y_preds = sigmoid(self.ypred_record) > 0.5
        score = accuracy_score(self.ytrue_record, y_preds)
        return score
    
    def recall_score(self):
        y_preds = sigmoid(self.ypred_record) > 0.5
        score = recall_score(self.ytrue_record, y_preds, zero_division=0)
        return score

    def precision_score(self):
        y_preds = sigmoid(self.ypred_record) > 0.5
        score = precision_score(self.ytrue_record, y_preds, zero_division=0)
        return score

    def specificity_score(self):
        y_preds = sigmoid(self.ypred_record) > 0.5
        score = specificity_score(self.ytrue_record, y_preds)
        return score

    def roc_auc_score(self) -> float:
        """
        Args:
            None
        Returns:
            score (float): AUC-ROC score.
        """
        y_preds = sigmoid(self.ypred_record)
        score = roc_auc_score(self.ytrue_record, y_preds)
        return score
    
    def count(self, target):
        y_preds = sigmoid(self.ypred_record) > 0.5        
        tn, fp, fn, tp = confusion_matrix(self.ytrue_record, y_preds).ravel()
        if target == "tp":
            return tp
        elif target == "fp":
            return fp
        elif target == "fn":
            return fn
        elif target == "tn":
            return tn
        else:
            raise
    
    def average_precision_score(self):
        y_preds = sigmoid(self.ypred_record)
        score = average_precision_score(
            self.ytrue_record, y_preds)
        return score

    def show_result(self) -> None:
        """
        Args:
            is_multilabel (bool): 
        Returns:
            None
        """
        y_preds = sigmoid(self.ypred_record)
        y_preds = (y_preds > 0.5).astype(int)
        # print("pred", len(y_preds), y_preds.sum())
        # print("gt", len(self.ytrue_record), np.unique(self.ytrue_record, return_counts=True))
        conf_matrix = confusion_matrix(self.ytrue_record, y_preds)
        print("Confusion Matrix")
        print(conf_matrix)
        print(f"Macro F1: {self.calc_f1():.4f}")
        print(f"AUC-ROC: {self.roc_auc_score():.4f}")
        print(f"Average Loss: {self.average_loss():.4f}")

class PretrainingMonitor:

    def __init__(
        self, 
        eval_every: int, 
        save_model_every: int,
        dump_every: int,
        total_samples: int
    ) -> None:

        self.eval_every = eval_every
        self.dump_every = dump_every
        self.save_model_every = save_model_every
        self.total_samples = total_samples

        self.n_samples_passed = 0
        self.n_samples_from_last_eval = 0
        self.n_samples_from_last_save = 0
        self.n_samples_from_last_dump = 0

    def update_counter(self, n_processed):

        self.n_samples_passed += n_processed
        self.n_samples_from_last_eval += n_processed
        self.n_samples_from_last_save += n_processed
        self.n_samples_from_last_dump += n_processed

    def trigger_eval(self):
        """
        Args:
            None
        Returns:
            is_eval (bool):
        """
        trigger_eval = self.n_samples_from_last_eval > self.eval_every
        if trigger_eval:
            print(
                f"{self.n_samples_passed} / {self.total_samples} "
                f"({self.n_samples_passed / self.total_samples * 100:.3f})"
            )
            self.n_samples_from_last_eval = 0
        return trigger_eval

    def trigger_saving(self):
        trigger_save = self.n_samples_from_last_save > self.save_model_every
        if trigger_save:
            print(
                "To save interim model. "
                f"{self.n_samples_passed} / {self.total_samples} "
                f"({self.n_samples_passed / self.total_samples * 100:.3f})"
            )
            self.n_samples_from_last_save = 0
        return trigger_save

    def trigger_dumping(self):
        trigger_dump = self.n_samples_from_last_dump > self.dump_every
        if trigger_dump:
            print(
                "To dump current result. "
                f"{self.n_samples_passed} / {self.total_samples} "
                f"({self.n_samples_passed / self.total_samples * 100:.3f})"
            )
            self.n_samples_from_last_dump = 0
        return trigger_dump

    def trigger_break(self):
        return self.n_samples_passed > self.total_samples

class Storer:

    def __init__(
        self, 
        save_dir: str, 
        store_interim_model: bool=False,
        store_keys: List=["total_loss", "score"]
    ):
        """
        Args:
            save_dir (str): Path to save dir
        Returns:
            None
        """
        self.save_samples = True # Initially True, becomes False after first save.

        os.makedirs(save_dir, exist_ok=True)
        if store_interim_model:
            os.makedirs(save_dir+"/interims", exist_ok=True)
        self.save_dir = save_dir

        self.trains = {key: {} for key in store_keys}
        self.evals = {key: {} for key in store_keys}

    def save_params(self, params) -> None:
        """
        Save parameters.
        Args:
            params
        Returns:
            None
        """
        # Save parameters as pickle file.
        savename = self.save_dir + "/params.pkl"
        with open(savename, "wb") as fp:
            pickle.dump(params, fp)

        # Save parameters as text file.
        savename = self.save_dir + "/params.txt"
        with open(savename, "w") as f:
            for k, v in vars(params).items():
                f.write(f"{k}: {v}\n")

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

    def _store(self, keeper, epoch, epoch_results):
        for key in epoch_results.keys():
            keeper[key][epoch] = epoch_results[key]
        return keeper

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
        print(epoch_result_dict)
        if is_eval:
            self._store(self.evals, epoch, epoch_result_dict)
            # self.evals["loss"][epoch] = epoch_result_dict["loss"]
            # self.evals["score"][epoch] = epoch_result_dict["score"]
        else:
            self._store(self.trains, epoch, epoch_result_dict)
            # self.trains["loss"][epoch] = epoch_result_dict["loss"]
            # self.trains["score"][epoch] = epoch_result_dict["score"]

    def store_logs(self, n_samples=None):
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
    
    def save_generated(
        self, 
        model, 
        n_sample, 
        duration,
        frequency,
        device,
        denom=1e6, 
        n_dump: int=5,
        is_best: bool=False,
        input_data=None
    ):
        """
        Args:

        Returns:
            None
        """
        get_recon = False if input_data is None else True

        # Generate samples or get reconstruction.
        model.eval()
        if not get_recon:
            z = torch.randn(n_dump, model.z_dim).to(device)
            generated = model.generate(z)
        else:
            generated = model.reconstruct(input_data)
        generated = generated.cpu().detach().numpy()
        del model

        if is_best:
            assert not get_recon
            save_loc = self.save_dir + f"/best_samples"
        else:
            # Prepare save location.
            if n_sample < denom:
                # calculate appropriate denominator based on n_sample.
                # eg. n_sample = 1500, denom = 1e2, n_sample = 150, denom=1e1
                denom = 10 ** (len(str(n_sample)) - 2)

            power = round(math.log(denom, 10), 3)
            n_sample_d = n_sample / denom
            
            info = f"{int(n_sample_d):06d}E{power:.2f}"
            if get_recon:
                save_loc = self.save_dir + f"/recons/p{info}"
            else:
                save_loc = self.save_dir + f"/dumps/p{info}"
        os.makedirs(save_loc, exist_ok=True)

        # Save generated samples.
        for i in range(n_dump):
            savename = save_loc + f"/dump_{i:02d}.png"
            # print(generated[i, 0].shape)
            make_ecg_plot(
                generated[i, 0], 
                duration,
                frequency,
                savename
            )

    def save_sample(
        self, 
        sample, 
        duration,
        frequency,
        n_samples=5
    ):
        """
        Args:

        Returns:

        """
        if not self.save_samples:
            return

        sample = sample.detach().cpu().numpy()
        save_loc = self.save_dir + f"/inputs"
        os.makedirs(save_loc, exist_ok=True)

        for i in range(n_samples):
            savename = save_loc + f"/sample_{i:02d}.png"
            make_ecg_plot(
                sample[i, 0], 
                duration, 
                frequency,
                savename
            )
        self.save_samples = False


class EarlyStopper:

    def __init__(self, mode: str, patience: int):
        """
        Args:
            mode (str): max or min
            patience (int):
        Returns:
            None
        """
        assert (mode in ["max", "min"])
        self.mode = mode

        self.patience = patience
        self.num_bad_count = 0

        if mode == "max":
            self.best = -1 * np.inf
        else:
            self.best = np.inf

    def stop_training(self, metric: float):
        """
        Args:
            metric (float):
        Returns:
            stop_train (bool):
        """
        if self.mode == "max":

            if metric <= self.best:
                self.num_bad_count += 1
            else:
                self.num_bad_count = 0
                self.best = metric

        else:

            if metric >= self.best:
                self.num_bad_count += 1
            else:
                self.num_bad_count = 0
                self.best = metric

        if self.num_bad_count > self.patience:
            stop_train = True
            print("Early stopping applied, stop training")
        else:
            stop_train = False
            print(f"Patience: {self.num_bad_count} / {self.patience} (Best: {self.best:.4f})")
        return stop_train