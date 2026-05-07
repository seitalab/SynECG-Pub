import os
from typing import Tuple, Optional
from argparse import Namespace

import torch
import torch.distributed as dist
import numpy as np
from optuna.trial import Trial

from codes.supports import utils
from codes.pretrain_model import ModelPretrainer
from codes.train_clf import ClassifierTrainer

torch.backends.cudnn.deterministic = True

def run_train(
    args: Namespace, 
    save_root: str,
    run_pretraining: bool=False,
    trial: Optional[Trial]=None,
    weight_file_dir: Optional[str]=None
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
    rank = getattr(args, "rank", 0)
    is_main_process = (
        (not getattr(args, "distributed", False)) or
        rank == 0
    )

    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Prepare result storing directories
    if getattr(args, "distributed", False):
        if is_main_process:
            timestamp = utils.get_timestamp()
            save_setting = f"{timestamp}-{args.host}"
        else:
            save_setting = None
        obj = [save_setting]
        dist.broadcast_object_list(obj, src=0)
        save_setting = obj[0]
    else:
        timestamp = utils.get_timestamp()
        save_setting = f"{timestamp}-{args.host}"

    save_dir = os.path.join(
        save_root, 
        save_setting
    )

    # Trainer prep
    trainer = ModelPretrainer(args, save_dir)
    trainer.set_model()
    
    if weight_file_dir is not None:
        weight_file = os.path.join(weight_file_dir, "net.pth")
        trainer.set_weight(weight_file, args.freeze)
 
    if is_main_process:
        print("Preparing dataloader ...")
    train_loader = trainer.prepare_dataloader(
        data_split="train",
        is_train=True,
    )
    valid_loader = trainer.prepare_dataloader(
        data_split="val",
        is_train=False,
    )

    trainer.set_optimizer()
    trainer.save_params()

    if is_main_process:
        print("Starting training ...")
    trainer.run(train_loader, valid_loader)
    _, best_result = trainer.get_best_loss()

    if getattr(args, "distributed", False):
        dist.barrier()

    del trainer

    # Return best validation loss when executing hyperparameter search.
    return best_result, save_dir

if __name__ == "__main__":

    pass
