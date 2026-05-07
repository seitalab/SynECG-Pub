import os
import sys
from typing import Tuple, Optional
from argparse import Namespace
from collections import Counter

import torch
import numpy as np
from optuna.trial import Trial

sys.path.append("..")
from codes import utils
from codes.train_model import ModelTrainer

torch.backends.cudnn.deterministic = True

def get_class_weight(class_weight, train_labels):
    if class_weight == "auto":
        weight = utils.calc_class_weight(train_labels)
    elif class_weight.startswith("manual-"):
        weight = np.array([float(class_weight[7:])])
    else:
        raise ValueError
    return weight

def run_train(
    params: Namespace, 
    save_root: str,
    trial: Optional[Trial]=None,
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
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)

    # Prepare result storing directories
    timestamp = utils.get_timestamp()
    save_setting = f"{timestamp}-{params.host}"
    save_dir = os.path.join(
        save_root, 
        save_setting
    )

    # Trainer prep
    trainer = ModelTrainer(params, save_dir)
    trainer.set_trial(trial)
    trainer.set_model()
    if params.finetune_target is not None:
        trainer.set_pretrained_model(params.freeze)
 
    print("Preparing dataloader ...")
    train_loader = trainer.prepare_dataloader(
        data_split="train",
        is_train=True,
    )
    valid_loader = trainer.prepare_dataloader(
        data_split="val",
        is_train=False,
    )

    if params.class_weight == "balanced":
        weight = Counter(train_loader.dataset.label)
    else:
        weight = None
    trainer.set_lossfunc(weight)
    
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