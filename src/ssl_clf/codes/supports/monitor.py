import os
import sys

import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, 
    roc_auc_score, 
    roc_curve, 
    confusion_matrix,
    accuracy_score, 
    average_precision_score,
    # multilabel_confusion_matrix, 
    recall_score, 
    precision_score,
)
# import matplotlib.pyplot as plt

sys.path.append("../utils")
from ecg_plot import make_ecg_plot

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Monitor:

    def __init__(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """
        self.num_data = 0
        self.total_loss = 0
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

    def store_loss(self, loss: float) -> None:
        """
        Args:
            loss (float): Mini batch loss value.
        Returns:
            None
        """
        self.total_loss += loss

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

    def average_loss(self) -> float:
        """
        Args:
            None
        Returns:
            average_loss (float):
        """
        return self.total_loss / self.num_data

    def macro_f1(self) -> float:
        """
        Args:
            None
        Returns:
            score (float): F1 score.
        """
        y_preds = sigmoid(self.ypred_record)
        y_preds = (y_preds > 0.5).astype(int)
        score = f1_score(self.ytrue_record, y_preds)
        return score

    def accuracy(self) -> float:
        """
        Args:
            None
        Returns:
            score (float):
        """            
        y_preds = sigmoid(self.ypred_record)
        y_preds = (y_preds > 0.5).astype(int)
        score = accuracy_score(self.ytrue_record, y_preds)
        return score

    def recall(self) -> float:
        """
        Args:
            None
        Returns:
            score (float):
        """
        y_preds = sigmoid(self.ypred_record)
        y_preds = (y_preds > 0.5).astype(int)
        score = recall_score(self.ytrue_record, y_preds)
        return score
    
    def precision(self) -> float:
        """
        Args:
            None
        Returns:
            score (float):
        """
        y_preds = sigmoid(self.ypred_record)
        y_preds = (y_preds > 0.5).astype(int)
        score = precision_score(
            self.ytrue_record, y_preds, zero_division=0)
        return score

    def auprc_score(self) -> float:
        """
        Args:
            None
        Returns:
            score (float): AUC-PRC score.
        """
        y_preds = sigmoid(self.ypred_record)
        y_preds = (y_preds > 0.5).astype(int)

        score = average_precision_score(self.ytrue_record, y_preds)
        return score

    def roc_auc_score(self) -> float:
        """
        Args:
            None
        Returns:
            score (float): AUC-ROC score.
        """
        y_preds = sigmoid(self.ypred_record)
        y_preds = (y_preds > 0.5).astype(int)

        score = roc_auc_score(self.ytrue_record, y_preds)
        return score

    def show_per_class_result(self) -> None:
        """
        Args:
            is_multilabel (bool): 
        Returns:
            None
        """
        y_preds = sigmoid(self.ypred_record)
        y_preds = (y_preds > 0.5).astype(int)
        conf_matrix = confusion_matrix(self.ytrue_record, y_preds)
        print("Confusion Matrix")
        print(conf_matrix)

    def confusion_matrix(self) -> np.ndarray:
        """
        Args:
            is_multilabel (bool): 
        Returns:
            None
        """
        y_preds = sigmoid(self.ypred_record)
        y_preds = (y_preds > 0.5).astype(int)
        conf_matrix = confusion_matrix(self.ytrue_record, y_preds)
        return conf_matrix

    def dump_errors(self, dump_loc, dump_type: str, n_dump: int=10):
        """
        Args:

        Returns:

        """
        duration = 10
        fs = 500

        y_preds = sigmoid(self.ypred_record)
        y_preds = (y_preds > 0.5).astype(int)
        if dump_type == "fp":
            false_positives = (self.ytrue_record == 0) & (y_preds == 1)
            targets = np.where(false_positives)[0]
        elif dump_type == "fn":
            false_negatives = (self.ytrue_record == 1) & (y_preds == 0)
            targets = np.where(false_negatives)[0]
        elif dump_type == "tp":
            true_positives = (self.ytrue_record == 1) & (y_preds == 1)
            targets = np.where(true_positives)[0]
        elif dump_type == "tn":
            true_negatives = (self.ytrue_record == 0) & (y_preds == 0)
            targets = np.where(true_negatives)[0]

        n_dump = min(len(targets), n_dump)
        if n_dump == 0:
            return
        
        idxs = np.random.choice(len(targets), n_dump)
        print(f"Storing {dump_type} samples ...")
        for idx in tqdm(idxs):
            input_idx = targets[idx]
            
            ecg = self.inputs[input_idx]
            savename = os.path.join(dump_loc, f"{dump_type}_{input_idx:08d}.png")
            # print(savename)
            make_ecg_plot(ecg, duration, fs, savename)

    # def store_sample(self, n_sample: int=10):
    #     """
    #     Args:

    #     Returns:

    #     """
    #     n_stored = len(self.ytrue_record)
    #     idxs = np.random.choice(n_stored, n_sample)
    #     for idx in idxs:
    #         # WIP
    #         fig, axs = plt.subplots(2, 1, figsize=(6, 8))

    #         # Plot on the first subplot
    #         axs[0].plot(x, y1, label='sin(x)')
    #         axs[0].set_title('First Plot')
    #         axs[0].set_xlabel('x')
    #         axs[0].set_ylabel('y')
    #         axs[0].legend()

    #         # Plot on the second subplot
    #         axs[1].plot(x, y2, label='cos(x)', color='orange')
    #         axs[1].set_title('Second Plot')
    #         axs[1].set_xlabel('x')
    #         axs[1].set_ylabel('y')
    #         axs[1].legend()

    #         # Adjust spacing between subplots
    #         plt.tight_layout()

    #         # Save the plots to a file (e.g., PNG format)
    #         plt.savefig('vertical_line_plots.png')


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