from argparse import Namespace
from typing import Iterable

import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from optuna.trial import Trial

from codes.supports.storer import Storer
from codes.data.dataloader import prepare_dataloader
from codes.models.model import prepare_model

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
        self.is_main_process = (
            (not getattr(args, "distributed", False)) or
            getattr(args, "rank", 0) == 0
        )
        self.storer = Storer(
            save_dir,
            hasattr(args, "save_model_every"),
            enabled=self.is_main_process,
        )
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

    def set_model(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """
        model = prepare_model(self.args)
        model = model.to(self.args.device)

        if getattr(self.args, "distributed", False):
            # SSSD contains optional/conditional paths and can leave some parameters
            # unused in a given iteration depending on the batch content.
            ddp_find_unused = getattr(self.args, "ddp_find_unused_parameters", True)
            if str(self.args.device).startswith("cuda"):
                model = DDP(
                    model,
                    device_ids=[getattr(self.args, "local_rank", 0)],
                    output_device=getattr(self.args, "local_rank", 0),
                    find_unused_parameters=ddp_find_unused,
                )
            else:
                model = DDP(
                    model,
                    find_unused_parameters=ddp_find_unused,
                )

        self.model = model

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
