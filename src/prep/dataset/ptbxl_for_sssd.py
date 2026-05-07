import os
import ast

import yaml
import wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from ptbxl import PTBXLPreparator

cfg_file = "../../../config.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)

class PTBXL_SSSD_DataPreparator(PTBXLPreparator):

    def __init__(self): 

        self.save_loc = os.path.join(
            cfg["settings"]["common"]["save_root"], 
            "PTBXL" + f"-sssd_data"
        )
        os.makedirs(self.save_loc, exist_ok=True)

        self._prep_ecg()

    def _prep_ecg(self):
        """
        Args:

        Returns:

        """
        df_target = pd.read_csv(
            cfg["settings"]["ptbxl"]["src"] + "/../ptbxl_database.csv"
        )
        df_label_list = pd.read_csv(
            cfg["settings"]["ptbxl"]["src"] + "/../scp_statements.csv",
            index_col=0
        )
        labels = list(df_label_list.index)

        ptbxl_ecgs, label_vectors = [], []
        for _, row in tqdm(df_target.iterrows(), total=len(df_target)):
            label_vector = np.zeros(len(labels))
            target_id = row["ecg_id"]
            label = ast.literal_eval(row["scp_codes"])
            for lb in label.keys():
                lb_idx = labels.index(lb)
                label_vector[lb_idx] = 1

            target_file = os.path.join(
                cfg["settings"]["ptbxl"]["src"], 
                f"{int(target_id/1000)*1000:05d}",
                f"{target_id:05d}_hr"
            )
            ecg_lead = wfdb.rdrecord(target_file).p_signal

            if len(ecg_lead) != 5000:
                continue
            
            # error if `nan` exists.
            assert not np.isnan(ecg_lead).any()

            # Make sure loading condition is same with 500Hz version.
            target_file = target_file.replace("records500/", "records100/")
            target_file = target_file.replace("_hr", "_lr")
            ecg_lead = wfdb.rdrecord(target_file).p_signal
            assert len(ecg_lead) == 1000
            assert not np.isnan(ecg_lead).any()

            ptbxl_ecgs.append(ecg_lead)
            label_vectors.append(label_vector)
        self.ecgs = np.array(ptbxl_ecgs)
        self.labels = np.array(label_vectors)
        assert len(self.ecgs) == len(self.labels)

    def _save_data(
        self, 
        data: np.ndarray, 
        labels: np.ndarray, 
        datatype: str, 
        seed: int=None
    ):
        """
        Args:

        Returns:

        """
        if seed is not None:
            fname = f"{datatype}_seed{seed:04d}.npy"
        else:
            fname = f"{datatype}.npy"
        
        savename = os.path.join(
            self.save_loc,
            fname
        )
        
        # Save as npy.
        np.save(savename, data)

        savename_lb = savename.replace(".npy", "_labels.npy")
        np.save(savename_lb, labels)

    def make_dataset(self):
        """
        Args:

        Returns:

        """
        idxs = np.arange(len(self.ecgs))
        tr_idx, te_idx = train_test_split(
            idxs, 
            test_size=cfg["split"]["test"]["size"], 
            random_state=cfg["split"]["test"]["seed"]
        )
        self._save_data(self.ecgs[te_idx], self.labels[te_idx], "test")

        seeds = cfg["split"]["train_val"]["seeds"]
        for i, seed in enumerate(seeds):
            print(f"{i+1}/{len(seeds)}")
            tr_sp_idx, v_sp_idx = train_test_split(
                tr_idx, 
                test_size=cfg["split"]["train_val"]["size"], 
                random_state=seed
            )
            Xtr_sp = self.ecgs[tr_sp_idx]
            ytr_sp = self.labels[tr_sp_idx]
            Xv_sp = self.ecgs[v_sp_idx]
            yv_sp = self.labels[v_sp_idx]
            self._save_data(Xtr_sp, ytr_sp, "train", seed)
            self._save_data(Xv_sp, yv_sp, "val", seed)
        print("Done")

if __name__ == "__main__":

    preparator = PTBXL_SSSD_DataPreparator()
    preparator.make_dataset()
    print("Done")
