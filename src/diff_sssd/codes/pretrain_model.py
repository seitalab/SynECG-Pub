import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional

from codes.train_base import BaseTrainer
from codes.supports.monitor import Monitor

class ModelPretrainer(BaseTrainer):

    def _set_device(self, X):
        """
        Args:
            X :
        Returns:
            None
        """
        if type(X) in [tuple, list]:
            return tuple(x.to(self.args.device).float() for x in X)
        return X.to(self.args.device).float()
    
    def _count_samples(self, mini_batch):
        """
        Args:
            mini_batch :
        Returns:
            count (int):
        """
        if type(mini_batch) in [tuple, list]:
            return len(mini_batch[0])
        return len(mini_batch)

    def _evaluate(
        self, 
        loader, 
        max_eval_sample: int=1000000000 # 1e9
    ) -> Dict:
        """
        Args:
            loader :
        Returns:
            result_dict (Dict):
        """
        monitor = Monitor()
        self.model.eval()

        n_evaled = 0
        with torch.no_grad():

            for X, y in tqdm(loader, disable=not self.is_main_process):
                
                # X = self._set_device(X)
                minibatch_size = self._count_samples(X)

                minibatch_loss = self.model(X, y)
                n_evaled += minibatch_size

                monitor.store_loss(float(minibatch_loss) * len(X))
                monitor.store_num_data(len(X))
                
                if max_eval_sample is not None:
                    if n_evaled > max_eval_sample:
                        break

        total_loss = monitor.total_loss
        total_num_data = monitor.num_data
        if getattr(self.args, "distributed", False):
            device = self.args.device
            total_loss_t = torch.tensor(total_loss, device=device, dtype=torch.float64)
            total_num_t = torch.tensor(total_num_data, device=device, dtype=torch.float64)
            dist.all_reduce(total_loss_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_num_t, op=dist.ReduceOp.SUM)
            total_loss = float(total_loss_t.item())
            total_num_data = max(float(total_num_t.item()), 1.0)

        result_dict = {
            "score": 0,
            "loss": total_loss / total_num_data,
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

        self.args.total_samples = int(self.args.batch_size * self.args.n_iters)
        self.args.eval_every = int(self.args.batch_size * self.args.iters_per_ckpt)
        self.args.save_model_every = int(self.args.batch_size * self.args.iters_per_ckpt)

        samples_per_epoch = len(train_loader.dataset)
        max_epochs = self.args.total_samples // samples_per_epoch + 1
        n_samples = 0
        n_iters = 0
        n_iters_from_last_eval = 0
        n_iters_from_last_save = 0
        eval_every_iters = int(self.args.iters_per_ckpt)
        save_every_iters = int(self.args.iters_per_ckpt)
        eval_at_epoch_end = getattr(self.args, "eval_at_epoch_end", True)
        if self.is_main_process:
            print(
                "Eval schedule: "
                f"eval_every={self.args.eval_every} samples "
                f"({eval_every_iters} iters), "
                f"save_every={self.args.save_model_every} samples "
                f"({save_every_iters} iters), "
                f"eval_at_epoch_end={eval_at_epoch_end}"
            )

        for ep in range(1, max_epochs + 1):
            did_eval_in_epoch = False
            if self.is_main_process:
                print("==========================")
                print(f"Epoch {ep} / {max_epochs}")

            if getattr(self.args, "distributed", False):
                if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(ep)
                if hasattr(valid_loader, "sampler") and hasattr(valid_loader.sampler, "set_epoch"):
                    valid_loader.sampler.set_epoch(ep)
            
            self.model.train()
            for X, y in tqdm(train_loader, disable=not self.is_main_process):

                # X = self._set_device(X)

                minibatch_loss = self.model(X, y)

                self.optimizer.zero_grad()
                minibatch_loss.backward()
                self.optimizer.step()

                minibatch_size = self._count_samples(X)
                n_iters += 1
                n_samples += minibatch_size
                n_iters_from_last_eval += 1
                n_iters_from_last_save += 1

                # Evaluate training progress.
                if n_iters_from_last_eval >= eval_every_iters:
                    if self.is_main_process:
                        print(
                            f"{n_samples} / {self.args.total_samples} "
                            f"({n_samples / self.args.total_samples * 100:.3f})"
                        )
                    self.model.eval()
                    self._monitor_progress(
                        n_samples, train_loader, valid_loader)
                    self.storer.store_logs()
                    did_eval_in_epoch = True
                    n_iters_from_last_eval = 0
                    self.model.train()

                # Store model.
                if n_iters_from_last_save >= save_every_iters:
                    if self.is_main_process:
                        print(
                            "Saving interim model. "
                            f"{n_samples} / {self.args.total_samples} "
                            f"({n_samples / self.args.total_samples * 100:.3f})"
                        )
                    self.storer.save_model_interim(
                        self.model, n_samples, n_iter=n_iters)
                    n_iters_from_last_save = 0
                
                if n_samples > self.args.total_samples:
                    break

            # Ensure validation runs even when eval_every is large relative to epoch size.
            if eval_at_epoch_end and (not did_eval_in_epoch) and n_samples > 0 and n_iters_from_last_eval > 0:
                if self.is_main_process:
                    print("Epoch-end evaluation and log flush ...")
                self.model.eval()
                self._monitor_progress(n_samples, train_loader, valid_loader)
                self.storer.store_logs()
                n_iters_from_last_eval = 0
                self.model.train()

            if n_samples > self.args.total_samples:
                break

        # Ensure we flush at least one evaluation/log record at the end of training.
        # This makes train_scores.json / eval_scores.json available even when the
        # run finishes before crossing eval_every.
        if n_samples > 0 and n_iters_from_last_eval > 0:
            if self.is_main_process:
                print("Final evaluation and log flush ...")
            self.model.eval()
            self._monitor_progress(n_samples, train_loader, valid_loader)
            self.storer.store_logs()

    def _monitor_progress(self, epoch, train_loader, valid_loader):

        train_set_result = self._evaluate(
            train_loader, max_eval_sample=250)
        self.storer.store_epoch_result(
            epoch, train_set_result, is_eval=False)

        val_set_result = self._evaluate(valid_loader)
        if self.mode == "max":
            monitor_target = val_set_result["score"]
        else:
            monitor_target = val_set_result["loss"]
        self.storer.store_epoch_result(
            epoch, val_set_result, is_eval=True)

        self._update_best_result(monitor_target, val_set_result)

    def _update_best_result(self, monitor_target, eval_result):
        """
        Args:

        Returns:
            None
        """
        
        if monitor_target * self.flip_val < self.best_val * self.flip_val:
            if self.is_main_process:
                print(f"Val metric improved {self.best_val:.4f} -> {monitor_target:.4f}")
            self.best_val = monitor_target
            self.best_result = eval_result
            self.storer.save_model(self.model, monitor_target)
        else:
            if self.is_main_process:
                print(f"Val metric did not improve. Current best {self.best_val:.4f}")
