import os
import pickle

import yaml
import numpy as np
from sklearn.model_selection import train_test_split

cfg_file = "../../../config.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)


class PrepBase:
    def _save_data(self, data: np.ndarray, datatype: str, seed: int=None):
        """
        Args:
            datatype (str): "train", "val", "test"
            seed (int, optional): train/val 固定 seed.
        """
        if seed is not None:
            fname = f"{datatype}_seed{seed:04d}.pkl"
        else:
            fname = f"{datatype}.pkl"

        savename = os.path.join(
            self.save_loc,
            fname
        )

        with open(savename, "wb") as fp:
            pickle.dump(data, fp)

    def make_dataset(self):
        """
        Split data with the same logic used by previous CardiallyPreparator:
        - create test once with split.test
        - create multiple train/val pairs from cfg["split"]["train_val"]["seeds"]
        """
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
