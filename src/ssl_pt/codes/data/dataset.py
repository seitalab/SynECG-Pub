import os
import pickle
from copy import copy
from glob import glob
from typing import Optional, List

import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, get_worker_info

# root = "/home/nonaka/git/ecg_pj/SynthesizedECG/data"
cfg_file = "../../config.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)

class ContrastiveLearningDataset:

    def __init__(self, base_dataset, transform):

        assert base_dataset.transform is None

        self.base_dataset = base_dataset
        self.transform = transform

        self.load_label = self.base_dataset.with_label
        self._worker_seeded = False

    def _maybe_seed_worker(self):
        worker = get_worker_info()
        if worker is None or self._worker_seeded:
            return
        worker_seed = int(worker.seed % (2**32))
        np.random.seed(worker_seed)
        if hasattr(self.transform, "set_worker_seed"):
            self.transform.set_worker_seed(worker_seed)
        elif hasattr(self.transform, "transforms"):
            for tf in self.transform.transforms:
                if hasattr(tf, "set_worker_seed"):
                    tf.set_worker_seed(worker_seed)
        self._worker_seeded = True

    def __getitem__(self, index):
        self._maybe_seed_worker()
        
        if self.load_label:
            view, label = self.base_dataset[index]
            info = {
                "data": view,
                # "freq": self.base_dataset.src_freq,
                "label": label
            }            
        else:
            info = {
                "data": self.base_dataset[index],
                # "freq": self.base_dataset.src_freq
            }            

        info1, info2 = info, copy(info)
        view1 = self.transform(info1)
        view2 = self.transform(info2)

        data1, data2 = view1["data"], view2["data"]
        if self.load_label:
            assert view1["label"] == view2["label"]
            return data1, data2, view1["label"]
        return (data1, data2)

    def __len__(self):
        return len(self.base_dataset)

class ECGDataset(Dataset):

    def __init__(
        self, 
        datatype: str,
        seed: int,
        pos: str,
        neg: str,
        dataset: str,
        data_lim: Optional[int]=None, 
        transform: Optional[List]=None
    ) -> None:
        """
        Args:
            root (str): Path to dataset directory.
            datatype (str): Dataset type to load (train, valid, test)
            seed (int): 
            pos (str): Positive label dataset.
            neg (str): Negative label dataset.
            data_lim (int): Total number of samples (load data_lim/2 samples from pos/neg dataset each).
                If data_lim = 1000: pos = 500, neg = 500.
                In case, number of samples in pos/neg dataset is below data_lim / 2, 
                total number of samples used will be less than data_lim.
            transform (List): List of transformations to be applied.
        """
        assert(datatype in ["train", "val", "test"])
        
        self.data_loc = cfg["experiment"]["path"]["data_root"]

        # Limit total number of dataset    
        self.data, self.label = [], []
        if dataset is not None:
            self.data_lim = data_lim #if data_lim is not None else data_lim
            data = self._load_data(datatype, seed, dataset)
            for d in data:
                self.data.append(d)
                self.label.append(0)
            self.with_label = False
        else:
            self.data_lim = data_lim // 2 if data_lim is not None else data_lim
            data = self._load_data(datatype, seed, neg)
            for d in data:
                self.data.append(d)
                self.label.append(0)
            data = self._load_data(datatype, seed, pos)
            for d in data:
                self.data.append(d)
                self.label.append(1)
            self.with_label = True

        print(f"Loaded {datatype} set: {len(self.data)} samples.")

        self.transform = transform
        self._worker_seeded = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        worker = get_worker_info()
        if worker is not None and not self._worker_seeded:
            worker_seed = int(worker.seed % (2**32))
            np.random.seed(worker_seed)
            if hasattr(self.transform, "set_worker_seed"):
                self.transform.set_worker_seed(worker_seed)
            elif hasattr(self.transform, "transforms"):
                for tf in self.transform.transforms:
                    if hasattr(tf, "set_worker_seed"):
                        tf.set_worker_seed(worker_seed)
            self._worker_seeded = True

        data = self.data[index]

        if self.transform:
            sample = {"data": data}
            sample = self.transform(sample)
            data = sample["data"]

        if self.with_label:
            return data, torch.Tensor([self.label[index]])
        return data

    def _load_data(self, datatype: str, seed: int, dirname: str) -> np.ndarray:
        """
        Load file of target datatype.
        Args:
            datatype (str)
        Returns:
            X (np.ndarray): Array of shape (num_samples, 12, sequence_length).
        """
        if datatype == "test":
            filename = "test.pkl"
        else:
            filename = f"{datatype}_seed{seed:04d}.pkl"

        if dirname.startswith("gen-"):
            target_info = dirname.split("-")[1]
            dgm_used = target_info.split("/")[0]
            data_ver = target_info.split("/")[1]
            data_loc = os.path.join(
                cfg["generatives"]["data"][dgm_used][data_ver],
                datatype,
                "samples"
            )
            targets = sorted(glob(data_loc + "/idx*.pkl"))
            ecg_data = []
            print("Loading generated data...")
            for target in tqdm(targets):
                with open(target, "rb") as fp:
                    ecg_data.append(pickle.load(fp))
            ecg_data = np.concatenate(ecg_data, axis=0)
            if len(ecg_data.shape) == 3: # 250403 added.
                ecg_data = ecg_data[:, 0] # n_data, 1, seqlen => n_data, seqlen

        else:
            target = os.path.join(
                self.data_loc,
                dirname, 
                filename
            )
            with open(target, "rb") as fp:
                ecg_data = pickle.load(fp)

        # Randomly shuffle data if real data with data lim.
        # if not dirname.startswith("syn_"):
        #     if self.data_lim is not None:
        #         np.random.seed(seed)
                # np.random.shuffle(ecg_data)
                

        if self.data_lim is not None:
            print(f"WARNING: LIMITING NUMBER OF SAMPLES: {len(ecg_data)} -> {self.data_lim}")
            if self.data_lim < 500000:
                # temporal: 251117
                if len(ecg_data) < self.data_lim:
                    print(f"WARNING: data size {len(ecg_data)} is smaller than data_lim {self.data_lim}.")
                    return ecg_data
                # This is probably faster for small datasize 
                # but process is killed if datasize is large.
                np.random.seed(seed)
                idxs = np.random.choice(
                    len(ecg_data), 
                    size=self.data_lim, 
                    replace=False
                )
                ecg_data_lim = np.array(ecg_data[idxs])
                del ecg_data
                return ecg_data_lim
            else:
                np.random.seed(seed)
                np.random.shuffle(ecg_data)
                return ecg_data[:self.data_lim]
        return ecg_data
    
if __name__ == "__main__":
    pass
