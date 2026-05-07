from enum import Enum
from argparse import Namespace
from typing import Type

from torchvision import transforms
from torch.utils.data import DataLoader

from codes.data.dataset import ECGDataset, ContrastiveLearningDataset
from codes.data.ecg_noise_augmentation import ECGNoiseAugmentation
from codes.data.transform_funcs import (
    ToTensor, 
    RandomMask,
    RandomShift, 
    ScaleECG,
    AlignLength,
    ECGNoiseTransform
)
CONTRASTIVE_LEARNING = [
    "simclr", "moco", "byol", "ibot", "dino"
]

class ScaleType(Enum):
    NONE = "none"
    PER_SAMPLE = "per_sample"


def prepare_preprocess(
    params: Namespace, 
    is_train: bool,
    is_finetune: bool
) -> Type[transforms.Compose]:
    """
    Prepare and compose transform functions.
    Args:
        params (Namespace): 
        is_train (bool): 
    Returns:
        composed
    """
    transformations = []
    target_len = int(params.max_duration * params.target_freq)

    use_ecg_noise_aug = bool(getattr(params, "use_ecg_noise_aug", False))
    apply_noise = (
        use_ecg_noise_aug
        and is_train
        and (not is_finetune)
        and params.ssl == "mae"
    )

    if apply_noise:
        # Apply mV-based noise before normalization so range settings stay meaningful.
        transformations.append(AlignLength(target_len))
        noise_mode = getattr(params, "ecg_noise_mode", "combined")
        noise_seed = getattr(params, "ecg_noise_seed", None)
        if noise_seed is None and hasattr(params, "seed"):
            noise_seed = int(params.seed)
        elif noise_seed is not None:
            noise_seed = int(noise_seed)
        noise_augmenter = ECGNoiseAugmentation(
            sample_rate=int(params.target_freq),
            mode=noise_mode,
            seed=noise_seed,
        )
        transformations.append(
            ECGNoiseTransform(noise_augmenter, seed=noise_seed)
        )
        transformations.append(ScaleECG())
    else:
        # Keep the original preprocessing order when noise is disabled.
        transformations.append(ScaleECG())
        transformations.append(AlignLength(target_len))

    # Simple augmentations.
    if is_train:
        if is_finetune:
            transformations.append(RandomMask(params.aug_mask_ratio))
            transformations.append(RandomShift(params.max_shift_ratio))
        else:
            if params.ssl != "mae":
                transformations.append(RandomMask(params.aug_mask_ratio))
                transformations.append(RandomShift(params.max_shift_ratio))
            else:
                pass # No additional augmentation for mae.

    # ToTensor and compose.
    transformations.append(ToTensor())
    composed = transforms.Compose(transformations)
    return composed

def prepare_dataloader(
    params: Namespace,
    datatype: str,
    is_train: bool,
    is_finetune: bool=False
) -> Type[DataLoader]:

    transformations = prepare_preprocess(params, is_train, is_finetune)
    
    data_lim = params.data_lim if is_train else params.val_lim 
    if params.dataset.find("//") != -1:
        target_dataset = None
        pos_dataset = params.dataset.split("//")[0]
        neg_dataset = params.dataset.split("//")[1]
    elif params.dataset is not None:
        target_dataset = params.dataset
        pos_dataset = None
        neg_dataset = None
    else:
        target_dataset = None
        pos_dataset = params.pos_dataset
        neg_dataset = params.neg_dataset


    # Check if the ssl mode is contrastive learning.
    if params.ssl in CONTRASTIVE_LEARNING:
        is_contrastive_learning = True
    else:
        is_contrastive_learning = False
        
    # For contrastive learning, we do not apply the transformation to the dataset.
    transform_base = transformations if not is_contrastive_learning else None

    dataset = ECGDataset(
        datatype, 
        params.seed, 
        pos_dataset,
        neg_dataset,
        target_dataset,
        data_lim,
        transform_base
    )
    if is_contrastive_learning:
        dataset = ContrastiveLearningDataset(
            dataset, transformations
        )
        

    if not is_train:
        drop_last = False
    else:
        if params.data_lim is not None:
            data_lim = params.data_lim
        else:
            data_lim = 1e10
        if params.batch_size > data_lim:
            drop_last = False
        else:
            drop_last = True


    loader = DataLoader(
        dataset, 
        batch_size=params.batch_size, 
        shuffle=is_train, 
        drop_last=drop_last, 
        num_workers=params.n_workers
    )
    return loader
