from argparse import Namespace

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from codes.data.dataset import ECGDataset
from codes.data.transform_funcs import (
    ToTensor, 
    RandomMask,
    RandomShift, 
    ScaleECG,
    AlignLength
)

def prepare_preprocess(
    params: Namespace, 
    is_train: bool,
):
    """
    Prepare and compose transform functions.
    Args:
        params (Namespace): 
        is_train (bool): 
    Returns:
        composed
    """
    transformations = []
    transformations.append(ScaleECG())
    transformations.append(
        AlignLength(int(params.max_duration * params.target_freq))
    )

    # Simple augmentations.
    if is_train:
        transformations.append(RandomMask(params.mask_ratio))
        transformations.append(RandomShift(params.max_shift_ratio))

    # ToTensor and compose.
    transformations.append(ToTensor())
    composed = transforms.Compose(transformations)
    return composed

class DatasetFactory:
    @staticmethod
    def create_dataset(
        params: Namespace,
        datatype: str,
        is_train: bool, 
        transform
    ) -> Dataset:
        data_lim = params.data_lim if is_train else params.val_lim

        dataset = ECGDataset(
            datatype,
            seed=1,
            dataset=params.dataset,
            data_lim=data_lim,
            transform=transform,
        )

        return dataset

def prepare_dataloader(
    params: Namespace, 
    datatype: str,
    is_train: bool, 
) -> DataLoader:

    # Prepare transformation and dataset.
    transform = prepare_preprocess(params, is_train)
    dataset = DatasetFactory.create_dataset(
        params, datatype, is_train, transform)
    
    drop_last = \
        is_train and \
        (params.data_lim is None or params.batch_size <= params.data_lim)

    return DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=is_train,
        drop_last=drop_last,
        num_workers=params.n_workers
    )