import os
from typing import Tuple
from argparse import Namespace

import torch
import numpy as np

from codes.utils import utils
from codes.trainer import Trainer
from codes.gan_trainer import GANTrainer

torch.backends.cudnn.deterministic = True

def run_train(
    args: Namespace, 
    save_root: str,
) -> Tuple[float, str]:
    """
    Execute train code for ecg classifier
    Args:
        args (Namespace): Namespace for parameters used.
        save_root (str): 
    Returns:
        best_val_loss (float): 
        save_dir (str):
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Prepare result storing directories
    timestamp = utils.get_timestamp()
    save_setting = f"{timestamp}-{args.host}"
    save_dir = os.path.join(
        save_root, 
        save_setting
    )

    # Trainer prep
    if args.is_gan:
        trainer = GANTrainer(args, save_dir)
    else:
        trainer = Trainer(args, save_dir)
    trainer.set_model()
     
    print("Preparing dataloader ...")
    train_loader = trainer.prepare_dataloader(
        datatype="train",
        is_train=True,
    )
    valid_loader = trainer.prepare_dataloader(
        datatype="val",
        is_train=False,
    )

    trainer.set_optimizer()
    trainer.save_params()

    print("Starting training ...")
    trainer.run(train_loader, valid_loader)
    _, best_result = trainer.get_best_loss()

    del trainer

    # Return best validation loss when executing hyperparameter search.
    return best_result, save_dir

if __name__ == "__main__":

    pass