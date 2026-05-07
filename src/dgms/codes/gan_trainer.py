from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

# from codes.models.model import prepare_model
# from codes.data.dataloader import prepare_dataloader
from codes.utils.monitor import Monitor, PretrainingMonitor
from codes.trainer import Trainer

class GANTrainer(Trainer):

    monitor_target_keys = ["total_loss", "g_loss", "d_loss"]

    def set_optimizer(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """
        assert self.model is not None

        if self.args.optimizer == "adam":
            self.optimizer_d = optim.Adam(
                self.model.discriminator.parameters(), 
                lr=self.args.learning_rate_d
            )
            self.optimizer_g = optim.Adam(
                self.model.generator.parameters(), 
                lr=self.args.learning_rate_g
            )
        else:
            raise NotImplementedError

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

                d_loss, g_loss = self.model(X, update_discriminator=True)
                minibatch_loss = g_loss + d_loss

                monitor.store_loss(float(minibatch_loss) * len(X))
                monitor.store_loss(float(g_loss) * len(X), "g_loss")
                monitor.store_loss(float(d_loss) * len(X), "d_loss")
                monitor.store_num_data(len(X))

        result_dict = {
            "score": 0,
            "total_loss": monitor.average_loss(),
            "g_loss": monitor.average_loss("g_loss"),
            "d_loss": monitor.average_loss("d_loss"),
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

            for i, (X, _) in enumerate(train_loader):
                update_discriminator = i % self.args.update_disriminator_every == 0

                self.optimizer_d.zero_grad()
                self.optimizer_g.zero_grad()
                # X = (X * mask).to(self.args.device).float()
                X = X.to(self.args.device).float()

                d_loss, g_loss = self.model(X, update_discriminator)
                g_loss.backward()
                self.optimizer_g.step()
                if d_loss is not None:
                    d_loss.backward()
                    self.optimizer_d.step()

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
                    self.storer.save_generated(
                        self.model, 
                        monitor.n_samples_passed,
                        self.args.max_duration,
                        self.args.target_freq,
                        self.args.device
                    )

                if monitor.trigger_break():                    
                    break

            if monitor.trigger_break():
                break
