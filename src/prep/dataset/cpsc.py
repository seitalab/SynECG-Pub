import os
import pickle
from glob import glob

import yaml
import numpy as np
import pandas as pd
from scipy import io
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from prep_base import PrepBase

cfg_file = "../../../config.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)


class CPSCPreparator(PrepBase):

    def __init__(self, target_dx: str):

        self.target_dx = target_dx
        self.lead_idx = cfg["settings"]["cpsc"]["lead_idx"]

        self.save_loc = os.path.join(
            cfg["settings"]["common"]["save_root"], 
            "CPSC" + f"-{target_dx}"
        )
        os.makedirs(self.save_loc, exist_ok=True)

        self.files, labels = self._load_reference()

        if self.target_dx != "ALL":
            # update `self.files` by filtering non-target.
            self._prep_dxs(labels)
        self._prep_ecg()

    def _load_reference(self):
        """
        Args:

        Returns:
            files (np.ndarray): Array of shape (n_files,)
            labels (np.ndarray): Array of shape (n_files, 9).
        """

        reference_file = os.path.join(
            cfg["settings"]["cpsc"]["src"],
            cfg["settings"]["cpsc"]["reference"]
        )
        df_refs = pd.read_csv(reference_file)
        files = df_refs.Recording.values
        labels = self._process_label(df_refs)
        return files, labels

    def _process_label(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convert dataframe of label to np.ndarray of shape [num_labels = 9].
        Args:
            df (pd.DataFrame): DataFrame of labels (contains maximum of 3 labels.)
        Returns:
            labels (np.ndarray): Array of shape [n_samples, num_labels = 9], each element is a binary.
        """
        labels = np.zeros([df.shape[0], 9])

        labels_first = df.First_label.values
        labels_second = df.Second_label.values
        labels_third = df.Third_label.values

        # labels = [1 - 9] -> label_index = [0 - 8]
        for i in range(df.shape[0]):
            # Validate: normal labeled sample has no other labels.
            if labels_first[i] == 1:
                assert(np.isnan(labels_second[i]) \
                       and np.isnan(labels_third[i]))

            labels[i, labels_first[i] - 1] = 1
            if not np.isnan(labels_second[i]):
                labels[i, int(labels_second[i]) - 1] = 1
            if not np.isnan(labels_third[i]):
                labels[i, int(labels_third[i]) - 1] = 1
        return labels

    def _prep_dxs(self, labels):

        is_target = []

        dx_index = cfg["settings"]["cpsc"]["dx_to_index"][self.target_dx]
        for label in labels:
            is_target.append(label[dx_index] == 1)
        self.is_target = is_target

    def _open_ecg_files(self, target_file: str) -> np.ndarray:
        """
        Args:
            target_file (str): target file
        Returns:
            signal (ndarray): numpy array of selected lead
        """
        target_path = os.path.join(
            cfg["settings"]["cpsc"]["src"],
            "*", 
            target_file+".mat"
        )
        mat_file = glob(target_path)[0]
        record = io.loadmat(mat_file) # [`Sex`, `Age`, `ECG`]
        data = record["ECG"][0][0][2] # Array (SEX, AGE, ECG_signal)
        
        # (num_lead = 12, seqlen) -> (seqlen)
        signal = data[cfg["settings"]["cpsc"]["lead_idx"]] 
        return signal

    def _prep_ecg(self):

        cpsc_ecgs = []

        for file in tqdm(self.files):
            ecg = self._open_ecg_files(file)
            cpsc_ecgs.append(ecg)
        self.ecgs = np.array(cpsc_ecgs, dtype=object)


    def make_dataset(self):
        """
        Args:

        Returns:

        """
        self.ecgs = self.ecgs[self.is_target]

        Xtr, Xte = train_test_split(
            self.ecgs, 
            test_size=cfg["split"]["test"]["size"], 
            random_state=cfg["split"]["test"]["seed"]
        )
        self._save_data(Xte, "test")

        seeds = cfg["split"]["train_val"]["seeds"]
        for i, seed in enumerate(seeds):
            print(f"{i+1}/{len(seeds)}")
            Xtr_sp, Xv_sp = train_test_split(
                Xtr, 
                test_size=cfg["split"]["train_val"]["size"], 
                random_state=seed
            )
            self._save_data(Xtr_sp, "train", seed)
            self._save_data(Xv_sp, "val", seed)
        print("Done")


if __name__ == "__main__":

    # target_dxs = ["VPB", "NormalSinus", "Afib"]
    # target_dxs = ["ALL"]
    target_dxs = [
        "NORM", "AF", "IAVB", "PAC", "PVC", "RBBB", "STD"
    ]
    for target_dx in target_dxs:
        print(target_dx)
        preparator = CPSCPreparator(target_dx)
        preparator.make_dataset()
    print("Done")
