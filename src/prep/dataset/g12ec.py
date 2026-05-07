import os
from glob import glob

import yaml
import wfdb
import numpy as np
from tqdm import tqdm

from prep_base import PrepBase

cfg_file = "../../../config.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)

class G12ECPreparator(PrepBase):

    def __init__(self, target_dx: str):

        self.target_dx = target_dx
        self.target_code = str(cfg["settings"]["g12ec"]["dx_to_code"][target_dx])
        self.lead_idx = cfg["settings"]["g12ec"]["lead_idx"]

        self.save_loc = os.path.join(
            cfg["settings"]["common"]["save_root"], 
            "G12EC" + f"-{target_dx}"
        )
        os.makedirs(self.save_loc, exist_ok=True)

        self.targets = sorted(glob(cfg["settings"]["g12ec"]["src"] + "/*.hea"))

        if self.target_dx != "ALL":
            self._prep_dxs()
        self._prep_ecg()

    def _prep_dxs(self):
        """
        Args:

        Returns:

        """
        dxs = []
        for t in tqdm(self.targets):
            data = wfdb.rdrecord(t[:-4]) # without extension
            dxs.append(data.comments[2])
        self.dxs = dxs

    def _prep_ecg(self):
        """
        Args:

        Returns:

        """
        if self.target_dx != "ALL":
            is_target = np.array([
                e.find(self.target_code) != -1 for e in self.dxs
            ])
        else:
            # all true.
            is_target = np.ones(len(self.targets)).astype("bool")

        g12ec_ecgs = []
        for idx in tqdm(np.where(is_target)[0]):
            data = wfdb.rdrecord(self.targets[idx][:-4])

            ecg_ii = data.p_signal[:, self.lead_idx]
            assert data.sig_name[self.lead_idx] == "II"

            if len(ecg_ii) != 5000:
                continue

            if np.isnan(ecg_ii).any():
                continue

            g12ec_ecgs.append(ecg_ii)
        self.ecgs = np.array(g12ec_ecgs)
    
if __name__ == "__main__":

    # target_dxs = ["VPB", "NormalSinus", "Afib"]
    # target_dxs = ["ALL"]
    # target_dxs = ["LVH", "IRBBB", "IAVB", "PAC", "RBBB"]
    target_dxs = ["VPB"]
    for target_dx in target_dxs:
        print(target_dx)
        preparator = G12ECPreparator(target_dx)
        preparator.make_dataset()
    print("Done")
