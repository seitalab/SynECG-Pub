import os
from glob import glob
from typing import Dict

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from optuna import TrialPruned

from codes.models.prepare_model import prepare_clf_model
from codes.train_base import BaseTrainer
from codes.supports.monitor import Monitor, EarlyStopper
from codes.utils import aggregator
from codes.supports.set_weight import get_weight_file, set_weight

class ModelTrainer(BaseTrainer):

    def set_model(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """

        model, ssl = prepare_clf_model(self.args)
        model = model.to(self.args.device)

        self.model = model
        self.ssl = ssl

    def set_pretrained_model(
        self, 
        freeze: bool=False,
    ):
        """
        Set trained weight to model.
        Args:
            freeze (bool): Freeze parameter or not.
        Returns:
            None
        """
        assert (self.model is not None)

        # Load weight file.
        ft_target = self.args.finetune_target
        if ft_target.startswith("pt-extra"):
            weight_file = get_weight_file(ft_target, "extra")
        elif ft_target.startswith("progress-pt"):
            weight_file = get_weight_file(ft_target, "progress")
        elif ft_target[6:] in ["gru", "resnet"]:
            pt_id = int(ft_target[2:6]) # ptXXXX -> XXXX
            weight_file = get_weight_file(pt_id)
        elif ft_target.startswith("random-init"):
            weight_file = None
            
        else: 
            pt_id = int(ft_target[2:]) # ptXXXX -> XXXX
            weight_file = get_weight_file(pt_id)

        # Set weight.
        self.model.backbone.to("cpu")
        if not self.args.finetune_target == "random-init":
            self.model = set_weight(self.model, weight_file, self.ssl)

        # Freeze parameter.
        if freeze:
            for p in self.model.backbone.parameters():
                p.requires_grad = False

        # Move to device.
        self.model.backbone.to(self.args.device)
        self.model.to(self.args.device)

    # def _is_skip_param(self, arch, param_key):
    #     """
    #     Args:

    #     Returns:

    #     """
    #     pop = False

    #     if param_key.startswith("head."):
    #         pop = True
    #     if arch == "mae":
    #         if param_key.startswith("fc."):
    #             pop = True

    #     if arch == "transformer":
    #         if param_key.startswith("fc."):
    #             pop = True
    #         elif param_key.startswith("foot."):
    #             pop = True

    #     if arch == "mega":
    #         if param_key.startswith("fc."):
    #             pop = True
    #         elif param_key.startswith("foot."):
    #             pop = True

    #     if arch == "luna":
    #         if param_key.startswith("fc."):
    #             pop = True
    #         elif param_key.startswith("foot."):
    #             pop = True

    #     if arch == "embgru":
    #         if param_key.startswith("fc."):
    #             pop = True
    #         elif param_key.startswith("foot."):
    #             pop = True

    #     if arch == "emblstm":
    #         if param_key.startswith("fc."):
    #             pop = True
    #         elif param_key.startswith("foot."):
    #             pop = True

    #     if arch == "s4":
    #         if param_key.startswith("fc."):
    #             pop = True
    #         elif param_key.startswith("foot."):
    #             pop = True

    #     if arch == "effnetb0":
    #         if param_key.startswith("fc."):
    #             pop = True
    #         elif param_key.startswith("foot."):
    #             pop = True

    #     return pop

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

        monitor.show_per_class_result()
        result_dict = {
            "score": monitor.macro_f1(), 
            "loss": monitor.average_loss(),
            "y_trues": monitor.ytrue_record,
            "y_preds": monitor.ypred_record
        }
        return result_dict
        
    def _evaluate(
        self, 
        loader, 
        dump_errors: bool=False
    ) -> Dict:
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

                if self.args.neg_dataset.startswith("CPSC"):
                    pred_y = aggregator(self.model, X)
                else:
                    pred_y = self.model(X)

                minibatch_loss = self.loss_fn(pred_y, y)

                monitor.store_loss(float(minibatch_loss) * len(X))
                monitor.store_num_data(len(X))
                monitor.store_result(y, pred_y)
                if dump_errors:
                    monitor.store_input(X)

        monitor.show_per_class_result()

        if dump_errors:
            monitor.dump_errors(self.dump_loc, dump_type="fp")
            monitor.dump_errors(self.dump_loc, dump_type="fn")
            monitor.dump_errors(self.dump_loc, dump_type="tp")
            monitor.dump_errors(self.dump_loc, dump_type="tn")
        result_dict = {
            "score": monitor.macro_f1(),
            "loss": monitor.average_loss(),
            "y_trues": monitor.ytrue_record,
            "y_preds": monitor.ypred_record,
            "auroc": monitor.roc_auc_score(),
            "auprc": monitor.auprc_score(),
            "recall": monitor.recall(),
            "precision": monitor.precision(),
            "confusion_matrix": monitor.confusion_matrix()
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

    def _update_best_result(self, monitor_target, eval_result):
        """
        Args:

        Returns:
            None
        """
        
        if monitor_target * self.flip_val < self.best_val * self.flip_val:
            print(
                "Val metric improved:",
                f"{self.best_val:.4f} -> {monitor_target:.4f}"
            )
            self.best_val = monitor_target
            self.best_result = eval_result
            self.storer.save_model(self.model, monitor_target)
        else:
            print(
                "Val metric did not improve.",
                f"Current best {self.best_val:.4f}"
            )
