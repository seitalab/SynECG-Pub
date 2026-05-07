from argparse import Namespace
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from codes.train_base import BaseTrainer
from codes.supports.monitor import Storer, Monitor, EarlyStopper

class ClassifierTrainer(BaseTrainer):

    def __init__(
        self,
        args: Namespace,
        save_dir: str,
        mode: str="min"
    ) -> None:
        """
        Args:
            args (Namespace):
            save_dir (str): Directory to output model weights and parameter info.
            mode (str): 
        Returns:
            None
        """

        self.args = args

        self.storer = Storer(
            save_dir, hasattr(args, "save_model_every"))
        self.model = None
        
        assert mode in ["max", "min"]
        self.mode = mode
        self.flip_val = -1 if mode == "max" else 1

        self.best_result = None
        self.best_val = np.inf * self.flip_val # Overwritten during training.

    def set_trial(self, trial=None):
        self.trial = trial

    def set_lossfunc(self, weights=None) -> None:
        """
        Args:
            weights (Optional[np.ndarray]): 
        Returns:
            None
        """
        assert self.model is not None

        if weights is not None:
            weights = torch.Tensor(weights)
        
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights)
        self.loss_fn.to(self.args.device)

    def _train(self, loader) -> Dict:
        """
        Run train mode iteration.
        Args:
            loader:
        Returns:
            result_dict (Dict):
        """

        monitor = Monitor()
        self.model.train()

        for X, y in tqdm(loader):

            self.optimizer.zero_grad()
            X = X.to(self.args.device).float()
            y = y.to(self.args.device).float()
            pred_y = self.model(X)

            minibatch_loss = self.loss_fn(pred_y, y)

            minibatch_loss.backward()
            self.optimizer.step()

            monitor.store_loss(float(minibatch_loss) * len(X))
            monitor.store_num_data(len(X))
            monitor.store_result(y, pred_y)

        monitor.show_result()
        result_dict = {
            "score": monitor.calc_f1(), 
            "loss": monitor.average_loss(),
            "y_trues": monitor.ytrue_record,
            "y_preds": monitor.ypred_record
        }
        return result_dict
        
    def _evaluate(self, loader) -> Dict:
        """
        Args:
            loader :
        Returns:
            result_dict (Dict):
        """
        monitor = Monitor()
        self.model.eval()

        with torch.no_grad():

            for X, y in tqdm(loader):
                
                X = X.to(self.args.device).float()
                y = y.to(self.args.device).float()

                pred_y = self.model(X)

                minibatch_loss = self.loss_fn(pred_y, y)

                monitor.store_loss(float(minibatch_loss) * len(X))
                monitor.store_num_data(len(X))
                monitor.store_result(y, pred_y)

        monitor.show_result()

        result_dict = {
            "score": monitor.calc_f1(),
            "loss": monitor.average_loss(),
            "F1score": monitor.calc_f1(),
            "Accuracy": monitor.accuracy(),
            "Recall": monitor.recall_score(),
            "Precision": monitor.precision_score(),
            "Specificity": monitor.specificity_score(),
            "AUROC": monitor.roc_auc_score(),
            "AUPRC": monitor.average_precision_score(),
            "TP_count": monitor.count("tp"),
            "FP_count": monitor.count("fp"),
            "FN_count": monitor.count("fn"),
            "TN_count": monitor.count("tn"),
            "y_trues": monitor.ytrue_record,
            "y_preds": monitor.ypred_record,
            # "file_idxs": monitor.f_idxs
        }                   
        return result_dict

    def run(self, train_loader, valid_loader) -> None:
        """
        Args:
            train_loader (Iterable): Dataloader for training data.
            valid_loader (Iterable): Dataloader for validation data.
            mode (str): definition of best (min or max).
        Returns:
            None
        """
        self.best = np.inf * self.flip_val # Sufficiently large or small
        if self.trial is None:
            early_stopper = EarlyStopper(
                mode=self.mode, patience=self.args.patience)

        for epoch in range(1, self.args.epochs + 1):
            print("-" * 80)
            print(f"Epoch: {epoch:03d}")
            train_result = self._train(train_loader)
            self.storer.store_epoch_result(
                epoch, train_result, is_eval=False)

            if epoch % self.args.eval_every == 0:
                eval_result = self._evaluate(valid_loader)
                self.storer.store_epoch_result(
                    epoch, eval_result, is_eval=True)

                if self.mode == "max":
                    monitor_target = eval_result["score"]
                else:
                    monitor_target = eval_result["loss"]

                # Use pruning if hyperparameter search with optuna.
                # Use early stopping if not hyperparameter search (= trial is None).
                if self.trial is not None:
                    self.trial.report(monitor_target, epoch)
                    if self.trial.should_prune():
                        raise TrialPruned()
                else:
                    if early_stopper.stop_training(monitor_target):
                        break

                self._update_best_result(monitor_target, eval_result)

            self.storer.store_logs()