from argparse import Namespace
from typing import Iterable, Dict

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from codes.models.model import prepare_model
from codes.data.dataloader import prepare_dataloader
from codes.utils.monitor import Monitor, PretrainingMonitor, Storer

class Trainer:

    monitor_target_keys = ["total_loss", "ddpm_loss", "recon_loss"]

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
            save_dir, 
            hasattr(args, "save_model_every"),
            store_keys=self.monitor_target_keys + ["score"]
        )
        self.model = None
        
        
        assert mode in ["max", "min"]
        self.mode = mode
        self.flip_val = -1 if mode == "max" else 1

        self.best_result = None
        self.best_val = np.inf * self.flip_val # Overwritten during training.

    def prepare_dataloader(
        self, 
        datatype: str, 
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
            datatype, 
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

        if self.args.scheduler is not None:
            self._set_scheduler()

    def _set_scheduler(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """

        if self.args.scheduler.startswith("plateau-"):
            if self.args.scheduler == "plateau-01":
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, 
                    mode="min",
                    patience=10, 
                    factor=0.1
                )
            else:
                raise
        elif self.args.scheduler.startswith("cosine-"):
            if self.args.scheduler == "cosine-01":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=10, 
                    eta_min=0
                )
            else:
                raise
        elif self.args.scheduler.startswith("exp-"):
            if self.args.scheduler == "exp-01":
                self.scheduler = optim.lr_scheduler.ExponentialLR(
                    self.optimizer, 
                    gamma=0.9
                )
            else:
                raise
        elif self.args.scheduler.startswith("cyclic-"):
            if self.args.scheduler == "cyclic-01":
                self.scheduler = optim.lr_scheduler.CyclicLR(
                    self.optimizer, 
                    base_lr=self.args.learning_rate*0.1,
                    max_lr=self.args.learning_rate*2,
                )
            else:
                raise            
        else:
            raise

    def set_model(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """

        model = prepare_model(self.args)
        model = model.to(self.args.device)

        self.model = model

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

    def _update_best_result(self, monitor_target, eval_result):
        """
        Args:

        Returns:
            None
        """
        
        if monitor_target * self.flip_val < self.best_val * self.flip_val:
            print(f"Val metric improved {self.best_val:.4f} -> {monitor_target:.4f}")
            self.best_val = monitor_target
            self.best_result = eval_result
            self.storer.save_model(self.model, monitor_target)
            self.storer.save_generated(
                self.model, 
                None, # n_sample is not used.
                self.args.max_duration,
                self.args.target_freq,
                self.args.device,
                is_best=True
            )
        else:
            message = (
                f"Val metric did not improve ({monitor_target:.4f}). "
                f"Current best {self.best_val:.4f}"
            )
            print(message)    
  
    def _calc_max_epochs(self, train_loader):
        """
        Args:
            train_loader (Iterable):
        Returns:
            max_epochs (int):
        """
        samples_per_epoch = len(train_loader.dataset)
        max_epochs = self.args.total_samples // samples_per_epoch + 1
        return max_epochs

    def _evaluate(self, loader) -> Dict:
        """
        Args:
            loader :
        Returns:
            result_dict (Dict):
        """
        monitor = Monitor(self.monitor_target_keys)
        self.model.eval()

        with torch.no_grad():

            for X, _ in tqdm(loader):
                
                X = X.to(self.args.device).float()

                # Only save for the first batch.
                self.storer.save_sample(
                    X,
                    self.args.max_duration,
                    self.args.target_freq,
                )

                ddpm_loss, recon_loss = self.model(X, None)
                minibatch_loss = ddpm_loss + self.args.lambda_recon * recon_loss

                monitor.store_loss(float(minibatch_loss) * len(X))
                monitor.store_loss(float(ddpm_loss) * len(X), "ddpm_loss")
                monitor.store_loss(float(recon_loss) * len(X), "recon_loss")
                
                monitor.store_num_data(len(X))

        result_dict = {
            "score": 0,
            "total_loss": monitor.average_loss(),
            "ddpm_loss": monitor.average_loss("ddpm_loss"),
            "recon_loss": monitor.average_loss("recon_loss"),
        }
        return result_dict

    def run(self, train_loader, valid_loader):
        """
        Args:
            train_loader (Iterable): Dataloader for training data.
            valid_loader (Iterable): Dataloader for validation data.
            mode (str): definition of best (min or max).
        Returns:
            None
        """
        self.best = np.inf * self.flip_val # Sufficiently large or small

        max_epochs = self._calc_max_epochs(train_loader)
        monitor = PretrainingMonitor(
            self.args.eval_every, 
            self.args.save_model_every, 
            self.args.dump_every,
            self.args.total_samples
        )
        for _ in tqdm(range(max_epochs)):
            
            self.model.train()
            for X, mask in train_loader:

                self.optimizer.zero_grad()
                X = X.to(self.args.device).float() # X: bs, 1, seqlen
                mask = mask.to(self.args.device).float()

                ddpm_loss, recon_loss = self.model(X, mask)
                minibatch_loss = ddpm_loss + self.args.lambda_recon * recon_loss
                minibatch_loss.backward()

                if self.args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.grad_clip)

                self.optimizer.step()

                # Update counter.
                monitor.update_counter(len(X))

                # Evaluate training progress.
                if monitor.trigger_eval():
                    self.model.eval()
                    self._monitor_progress(
                        monitor.n_samples_passed, 
                        train_loader, 
                        valid_loader
                    )
                    self.storer.store_logs()                    
                    self.model.train()

                # Store model.
                if monitor.trigger_saving():
                    self.storer.save_model_interim(
                        self.model, monitor.n_samples_passed)

                if monitor.trigger_dumping():
                    self.model.eval()
                    # Dump generated data.
                    self.storer.save_generated(
                        self.model, 
                        monitor.n_samples_passed,
                        self.args.max_duration,
                        self.args.target_freq,
                        self.args.device
                    )
                    # Dump reconstruction data.
                    self.storer.save_generated(
                        self.model, 
                        monitor.n_samples_passed,
                        self.args.max_duration,
                        self.args.target_freq,
                        self.args.device,
                        input_data=X
                    )
                    self.model.train()

                if monitor.trigger_break():                    
                    break

            if monitor.trigger_break():
                break

    def _monitor_progress(self, n_samples, train_loader, valid_loader):

        train_set_result = self._evaluate(train_loader)
        self.storer.store_epoch_result(
            n_samples, train_set_result, is_eval=False)

        val_set_result = self._evaluate(valid_loader)
        if self.mode == "max":
            monitor_target = val_set_result["score"]
        else:
            monitor_target = val_set_result["total_loss"]
        self.storer.store_epoch_result(
            n_samples, val_set_result, is_eval=True)

        self._update_best_result(monitor_target, val_set_result)
