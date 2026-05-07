import os
import sys
from argparse import Namespace
from typing import Type

import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

_SSSD_STANDALONE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "sssd_standalone")
)
if _SSSD_STANDALONE_DIR not in sys.path:
    sys.path.insert(0, _SSSD_STANDALONE_DIR)
from sssd_standalone import PTBXLDataset

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)

DEFAULT_GIVEN_SPLIT_ROOT = os.path.join(
    _REPO_ROOT, "raw_data", "v251123_sssd_ptbxl", "Real"
)
DEFAULT_SELF_SPLIT_ROOT = os.path.join(
    _REPO_ROOT, "raw_data", "PTBXL-sssd_data"
)


def _resolve_data_roots(params: Namespace):
    exp_path_cfg = config.get("experiment", {}).get("path", {})
    cfg_data_root = exp_path_cfg.get("data_root", None)

    given_split_root = getattr(params, "given_split_root", None)
    if given_split_root is None:
        given_split_root = exp_path_cfg.get("diff_sssd_given_split_root", DEFAULT_GIVEN_SPLIT_ROOT)

    self_split_root = getattr(params, "self_split_root", None)
    if self_split_root is None:
        self_split_root = exp_path_cfg.get("diff_sssd_self_split_root", None)
    if self_split_root is None and cfg_data_root is not None:
        self_split_root = os.path.join(cfg_data_root, "PTBXL-sssd_data")
    if self_split_root is None:
        self_split_root = DEFAULT_SELF_SPLIT_ROOT

    return given_split_root, self_split_root

def prepare_path(data_split, dataset_type, given_split_root, self_split_root):
    if dataset_type == "given_split": # Official split
        d_path = os.path.join(given_split_root, f"{data_split}_ptbxl_1000.npy")
        if data_split == "val":
            # label -> val, data -> valid
            data_split = "valid"
        l_path = os.path.join(given_split_root, f"1000_{data_split}_labels.npy")
    elif dataset_type == "self_split": # Self split.
        if data_split == "test":
            data_split = "val"
        d_path = os.path.join(self_split_root, f"{data_split}_seed0007.npy")
        l_path = os.path.join(self_split_root, f"{data_split}_seed0007_labels.npy")
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    return d_path, l_path

def prepare_dataloader(
    params: Namespace,
    data_split: str,
    is_train: bool,
) -> Type[DataLoader]:

    given_split_root, self_split_root = _resolve_data_roots(params)
    d_path, l_path = prepare_path(
        data_split,
        params.dataset,
        given_split_root,
        self_split_root,
    )

    dataset = PTBXLDataset(
        data_path=d_path,
        labels_path=l_path,
        segment_length=1000
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

    use_distributed = getattr(params, "distributed", False)
    if use_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=getattr(params, "world_size", 1),
            rank=getattr(params, "rank", 0),
            shuffle=is_train,
            drop_last=drop_last,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = is_train

    loader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=params.n_workers,
        pin_memory=True,
    )
    return loader
