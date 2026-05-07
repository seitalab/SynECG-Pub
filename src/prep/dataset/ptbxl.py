import os
import ast

import yaml
import wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from prep_base import PrepBase

cfg_file = "../../../config.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)

class PTBXLPreparator(PrepBase):

    def __init__(self, target_dx: str, thres: float=100):

        self.target_dx = target_dx
        self.lead_idx = cfg["settings"]["ptbxl"]["lead_idx"]

        self.save_loc = os.path.join(
            cfg["settings"]["common"]["save_root"], 
            "PTBXL" + f"-{target_dx}"
        )
        os.makedirs(self.save_loc, exist_ok=True)

        self._prep_ecg(thres)

    def _prep_ecg(self, thres):
        """
        Args:

        Returns:

        """
        df = pd.read_csv(
            cfg["settings"]["ptbxl"]["src"] + "/../ptbxl_database.csv"
        )

        if self.target_dx == "ALL":
            df_target = df
        else:

            is_target = np.array([
                self.target_dx in ast.literal_eval(dx_dict).keys()
                for dx_dict in df.scp_codes.values
            ])
            df_dx = df[is_target]

            is_target = [
                ast.literal_eval(d)[self.target_dx] >= thres 
                for d in df_dx.scp_codes.values
            ]
            df_target = df_dx[is_target]

        ptbxl_ecgs = []
        for target_id in tqdm(df_target.ecg_id.values):
            target_file = os.path.join(
                cfg["settings"]["ptbxl"]["src"], 
                f"{int(target_id/1000)*1000:05d}",
                f"{target_id:05d}_hr"
            )
            ecg = wfdb.rdrecord(target_file)
            ecg_ii = ecg.p_signal[:, ecg.sig_name.index("II")]

            if len(ecg_ii) != 5000:
                continue
            
            # error if `nan` exists.
            assert not np.isnan(ecg_ii).any()

            ptbxl_ecgs.append(ecg_ii)
        self.ecgs = np.array(ptbxl_ecgs)

if __name__ == "__main__":

    L_Thres = ["AFIB", "PAC", "STD_", "ABQRS"]

    target_dxs = ["NORM", "AFIB", "PVC", "PAC", "AFLT", "WPW"]
    for target_dx in target_dxs:
        print(target_dx)
        thres = 0 if target_dx in L_Thres else 100
        preparator = PTBXLPreparator(target_dx, thres)
        preparator.make_dataset()
    print("Done")
