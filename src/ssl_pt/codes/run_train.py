import os
from typing import Tuple, Optional
from argparse import Namespace

import torch
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
    if run_pretraining:
        trainer = ModelPretrainer(args, save_dir)
    else:
        trainer = ClassifierTrainer(args, save_dir)
        trainer.set_trial(trial)
    trainer.set_model(load_ssl=run_pretraining)
    
    if weight_file_dir is not None:
        weight_file = os.path.join(weight_file_dir, "net.pth")
        trainer.set_weight(weight_file, args.freeze)
 
    print("Preparing dataloader ...")
    train_loader = trainer.prepare_dataloader(
        datatype="train",
        is_train=True,
        is_finetune=not run_pretraining
    )
    valid_loader = trainer.prepare_dataloader(
        datatype="val",
        is_train=False,
        is_finetune=not run_pretraining
    )

    if not run_pretraining:
        if args.class_weight == "auto":
            weight = utils.calc_class_weight(
                train_loader.dataset.label)
        elif args.class_weight.startswith("manual-"):
            weight = np.array([float(args.class_weight[7:])])
        else:
            raise ValueError
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