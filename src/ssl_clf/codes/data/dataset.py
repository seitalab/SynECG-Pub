import os
import pickle
from typing import Optional, List

import yaml
import torch
import numpy as np
from torch.utils.data import Dataset

# root = "/home/nonaka/git/ecg_pj/SynthesizedECG/data"
cfg_file = "../../config.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)

class ECGDataset(Dataset):

    def __init__(
        self, 
        datatype: str,
        seed: int,
        pos: str,
        neg: str,
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
            data_lim (int): Total number of samples used for each class.
                If data_lim = 1000: pos = 1000, neg = 1000.
                In case, number of samples in pos/neg dataset is below data_lim / 2, 
                total number of samples used will be less than data_lim.
            transform (List): List of transformations to be applied.
        """
        assert(datatype in ["train", "val", "test"])
        
        self.data_loc = cfg["experiment"]["path"]["data_root"]

        # Limit total number of dataset    
        self.data, self.label = [], []

        data_lim_pos, data_lim_neg = self._calc_data_lim(data_lim)
        data = self._load_data(datatype, seed, neg, data_lim_neg)
        for d in data:
            self.data.append(d)
            self.label.append(0)
        data = self._load_data(datatype, seed, pos, data_lim_pos)
        for d in data:
            self.data.append(d)
            self.label.append(1)

        print(f"Loaded {datatype} set: {len(self.data)} samples.")

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        if self.transform:
            sample = {"data": data}
            sample = self.transform(sample)
            data = sample["data"]
        return data, torch.Tensor([self.label[index]])

    def _calc_data_lim(self, data_lim):
        """
        if data_lim is None -> None, None
        if data_lim is <int_val>p -> int_val, None
        if data_lim is <int_val>m -> None, int_val
        if data_lim is <int_val> -> int_val, int_val

        Args:

        Returns:

        """
        if data_lim is None:
            return None, None
        
        if type(data_lim) == str:
            if data_lim.endswith("p"):
                return int(data_lim[:-1]), None
            elif data_lim.endswith("n"):
                return None, int(data_lim[:-1])
            else:
                raise
        return data_lim, data_lim

    def _load_data(
        self, 
        datatype: str, 
        seed: int, 
        dirname: str,
        data_lim: int
    ) -> np.ndarray:
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

        target = os.path.join(
            self.data_loc,
            dirname, 
            filename
        )
        with open(target, "rb") as fp:
            ecg_data = pickle.load(fp)

        if data_lim is not None:
            if len(ecg_data) < data_lim:
                return ecg_data

            print(f"WARNING: LIMITING NUMBER OF SAMPLES: {len(ecg_data)} -> {data_lim}")
            if data_lim < 500000:
                # This is probably faster for small datasize 
                # but process is killed if datasize is large.
                np.random.seed(seed)
                idxs = np.random.choice(
                    len(ecg_data), 
                    size=data_lim, 
                    replace=False
                )
                ecg_data_lim = np.array(ecg_data[idxs])
                del ecg_data
                return ecg_data_lim
            else:
                np.random.seed(seed)
                np.random.shuffle(ecg_data)
                return ecg_data[:data_lim]
        return ecg_data
    

class ECGDatasetWithDemographics(ECGDataset):

    def __init__(
        self, 
        datatype: str,
        seed: int,
        pos: str,
        neg: str,
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
            data_lim (int): Total number of samples used for each class.
                If data_lim = 1000: pos = 1000, neg = 1000.
                In case, number of samples in pos/neg dataset is below data_lim / 2, 
                total number of samples used will be less than data_lim.
            transform (List): List of transformations to be applied.
        """
        assert(datatype in ["train", "val", "test"])
        
        self.data_loc = cfg["experiment"]["path"]["data_root_demos"]
        assert data_lim is None # No data_lim when loading with demographic info.

        # Limit total number of dataset    
        self.data, self.demo, self.label = [], [], []

        data = self._load_data(datatype, seed, neg, load_demo=False)
        demo = self._load_data(datatype, seed, neg, load_demo=True)
        for da, dm in zip(data, demo):
            self.data.append(da)
            self.demo.append(dm)
            self.label.append(0)
        data = self._load_data(datatype, seed, pos, load_demo=False)
        demo = self._load_data(datatype, seed, pos, load_demo=True)
        for da, dm in zip(data, demo):
            self.data.append(da)
            self.demo.append(dm)
            self.label.append(1)

        self.demo = np.array(self.demo)
        self.label = np.array(self.label)
        assert len(self.data) == len(self.demo)
        assert len(self.data) == len(self.label)
        print(f"Loaded {datatype} set: {len(self.data)} samples.")

        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        demo = self.demo[index]

        if self.transform:
            sample = {"data": data}
            sample = self.transform(sample)
            data = sample["data"]
        return data, demo, torch.Tensor([self.label[index]])

    def _load_data(
        self, 
        datatype: str, 
        seed: int, 
        dirname: str,
        load_demo: bool=False
    ) -> np.ndarray:
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

        if load_demo:
            filename = filename.replace(".pkl", "_demo.pkl")

        target = os.path.join(
            self.data_loc,
            dirname, 
            filename
        )
        with open(target, "rb") as fp:
            target_data = pickle.load(fp)
        return target_data

if __name__ == "__main__":
    pass