import os
from typing import Tuple, Dict
from argparse import Namespace

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report

from codes.train_model import ModelTrainer
from codes.data.dataloader import (
    prepare_dataloader,
    prepare_dataloader_with_demographics
)

from codes.supports.utils import get_timestamp
from codes.supports.monitor import sigmoid, Monitor

class ModelEvaluator(ModelTrainer):

    def __init__(self, args: Namespace, dump_loc: str, device: str) -> None:
        """
        Args:
            args (Namespace):
            dump_loc (str):
            device (str):
        Returns:
            None
        """
        self.args = args
        self.args.device = device

        self.device = device
        self.model = None

        timestamp = get_timestamp()
        self.dump_loc = os.path.join(dump_loc, timestamp)

        os.makedirs(self.dump_loc, exist_ok=True)

    def set_weight(self, weight_file: str):
        """
        Set trained weight to model.
        Args:
            weight_file (str):
        Returns:
            None
        """
        assert (self.model is not None)

        self.model.to("cpu")
        if self.ssl == "byol":
            del self.model.backbone.online_projector
            del self.model.backbone.target_projector
            del self.model.backbone.predictor
        elif self.ssl == "ibot":
            del self.model.backbone.center
            del self.model.backbone.center_patch
            del self.model.backbone.teacher
            del self.model.backbone.proj_teacher


        # Temporal solution.
        state_dict = dict(torch.load(weight_file, map_location="cpu")) # OrderedDict -> dict

        old_keys = list(state_dict.keys())
        for key in old_keys:
            new_key = key.replace("module.", "")
            if key.find("mae.") != -1:
                new_key = key.replace("mae.", "backbone.")
            state_dict[new_key] = state_dict.pop(key)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)

    def run(self, loader, dump_errors=False) -> Dict:
        """
        Args:
            loader
        Returns:
            result_dict (Dict):
        """
        result_dict = self._evaluate(loader, dump_errors=dump_errors)
        report = classification_report(
            result_dict["y_trues"], 
            sigmoid(result_dict["y_preds"])>0.5, 
            digits=5, 
            zero_division=0.0
        )
        return result_dict, report

    def dump_target(self, eval_target: str):
        """
        Args:
            eval_target (str):
        Returns:
            None
        """
        with open(self.dump_loc + "/eval_target.txt", "w") as f:
            f.write(eval_target)

class ModelEvaluatorWithDemographics(ModelEvaluator):
    """
    ModelEvaluatorWithDemographics
    """
    
    def prepare_dataloader(self, data_split: str):

        # Prepare dataloader.
        loader = prepare_dataloader_with_demographics(
            self.args, 
            data_split
        )
        return loader

    def _evaluate(self, loader) -> Dict:
        """
        Args:
            loader :
        Returns:
            result_dict (Dict):
        """
        monitor = Monitor()
        self.model.eval()

        demo_info = []

        with torch.no_grad():

            for X, demo, y in tqdm(loader):
                
                X = X.to(self.args.device).float()
                y = y.to(self.args.device).float()
                demo_info.append(demo)

                # Must be PTBXL dataset.
                assert self.args.neg_dataset.startswith("PTBXL")
                pred_y = self.model(X)
                minibatch_loss = self.loss_fn(pred_y, y)

                monitor.store_loss(float(minibatch_loss) * len(X))
                monitor.store_num_data(len(X))
                monitor.store_result(y, pred_y)
        demo_info = np.concatenate(demo_info, axis=0)
        monitor.show_per_class_result()

        result_dict = {
            "score": monitor.macro_f1(),
            "loss": monitor.average_loss(),
            "y_trues": monitor.ytrue_record,
            "y_preds": monitor.ypred_record,
            "demo_info": demo_info,
            "auroc": monitor.roc_auc_score(),
            "auprc": monitor.auprc_score(),
            "recall": monitor.recall(),
            "precision": monitor.precision(),
            "confusion_matrix": monitor.confusion_matrix()
        }

        # demo info csv.
        df_result = self._make_result_dataframe(
            demo_info, 
            monitor.ytrue_record, 
            monitor.ypred_record
        )

        return result_dict, df_result      
    
    def _make_result_dataframe(self, demo_info, y_trues, y_preds):
        """
        Args:
            demo_info (list):
            y_trues (list):
            y_preds (list):
        Returns:
            None
        """
        assert len(demo_info) == len(y_trues)

        y_preds = sigmoid(y_preds)
        data = np.concatenate(
            [
                y_trues.reshape(-1, 1), 
                y_preds.reshape(-1, 1), 
                demo_info
            ], axis=1
        )
        df_result = pd.DataFrame(
            data, 
            columns=["y_true", "y_pred", "age", "sex"]
        )
        return df_result

    def run(self, loader, dump_errors=False) -> Tuple[Dict, pd.DataFrame]:
        """
        Args:
            loader
        Returns:
            result_dict (Dict):
            df_result (DataFrame):
        """
        return self._evaluate(loader)
