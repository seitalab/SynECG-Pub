from argparse import Namespace
from typing import Iterable

import torch.optim as optim
import numpy as np
from optuna.trial import Trial

from codes.supports.storer import Storer
from codes.data.dataloader import prepare_dataloader
# from codes.models.prepare_model import prepare_clf_model

class BaseTrainer:

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

        self.storer = Storer(save_dir)
        self.model = None
        
        
        assert mode in ["max", "min"]
        self.mode = mode
        self.flip_val = -1 if mode == "max" else 1

        self.best_result = None
        self.best_val = np.inf * self.flip_val # Overwritten during training.

    def prepare_dataloader(
        self, 
        data_split: str, 
        is_train: bool=False,
    ) -> Iterable:
        """
        Args:
            None
        Returns:
            loader (Iterable):
        """

        # Prepare dataloader.
        loader = prepare_dataloader(
            self.args, 
            data_split, 
            is_train=is_train, 
        )
        return loader

    def set_optimizer(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """
        assert self.model is not None

        if self.args.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.args.learning_rate
            )
        elif self.args.optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.model.parameters(), 
                lr=self.args.learning_rate
            )
        elif self.args.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.args.learning_rate
            )
        else:
            raise NotImplementedError
        
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, 
        #     patience=self.args.optimizer_patience, 
        #     factor=0.2
        # )

    # def set_model(self) -> None:
    #     """
    #     Args:
    #         None
    #     Returns:
    #         None
    #     """

    #     model = prepare_mae_model(self.args)
    #     model = model.to(self.args.device)

    #     self.model = model

    def set_trial(self, trial: Trial) -> None:
        """
        Args:
            trial (Trial): Optuna trial.
        Returns:
            None
        """
        self.trial = trial

    def save_params(self) -> None:
        """
        Save parameters.
        Args:
            params
        Returns:
            None
        """
        self.storer.save_params(self.args)

    def get_best_loss(self) -> float:
        """
        Args:
            None
        Returns:
            best_value (float):
        """
        return self.best_val, self.best_result

    def _train(self, iterator: Iterable):
        raise NotImplementedError

    def _evaluate(self, iterator: Iterable):
        raise NotImplementedError

    def run(self, train_loader: Iterable, valid_loader: Iterable):
        raise NotImplementedError