import os
import pickle
from typing import Dict
from argparse import Namespace

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from scipy.special import softmax

from codes.eval_model import (
    ModelEvaluator,
    ModelEvaluatorWithDemographics
)

def get_confmat_text(ytrues, ypreds):
    confmat = confusion_matrix(
        ytrues, 
        softmax(ypreds, axis=1).argmax(axis=1) # Need to fix here. Doesnot affect the result.
    )
    return np.array2string(confmat, separator=', ')

class ReportManager:

    def __init__(self, eval_target: str):

        self.report = "\n\nEVAL TARGET\n" + eval_target + "\n\n"

    def add_row(self, key, content, n_rep: int=2):

        self.report += f"{key}: {content}"

        for _ in range(n_rep):
            self.report += "\n"

    def get_report(self):

        return self.report.strip()

def run_eval(
    eval_target: str,
    device: str,
    dump_loc: str,
    multiseed_run: bool=True,
    dump_errors: bool=False,
    overwrite_params: dict=None
):
    report_manager = ReportManager(eval_target)

    if multiseed_run:
        dump_loc = os.path.join(dump_loc, "multirun", "eval")

    # Settings
    param_file = os.path.join(eval_target, "params.pkl")
    weightfile = os.path.join(eval_target, "net.pth")

    with open(param_file, "rb") as fp:
        params = pickle.load(fp)
    params.data_lim = None
    if overwrite_params is not None:
        params = vars(params)
        for key, val in overwrite_params.items():
            params[key] = val
        params = Namespace(**params)

    report_manager.add_row("Model", params.finetune_target)
    report_manager.add_row("Parameters", str(params))
    
    # Evaluator
    evaluator = ModelEvaluator(
        params, dump_loc, device)
    evaluator.set_model()
    evaluator.set_lossfunc()
    evaluator.set_weight(weightfile)

    loader = evaluator.prepare_dataloader(
        "val", is_train=False)
    val_result, val_report = evaluator.run(loader)
    val_confmat = get_confmat_text(
        val_result["y_trues"], 
        val_result["y_preds"]
    )

    # if params.dataset == "syn":
    #     test_set = "val"
    # else:
    #     test_set = "test"
    test_set = "test"
    loader = evaluator.prepare_dataloader(
        test_set, is_train=False)
    test_result, test_report = evaluator.run(
        loader, dump_errors=dump_errors)
    test_confmat = get_confmat_text(
        test_result["y_trues"], test_result["y_preds"]
    )

    report_manager.add_row(
        "Validation set result\n", val_report)
    report_manager.add_row(
        "Validation set confusion matrix\n", 
        val_confmat
    )
    report_manager.add_row(
        "Test set result\n", test_report)
    report_manager.add_row(
        "Test set confusion matrix\n", 
        test_confmat
    )
    report = report_manager.get_report()
    with open(os.path.join(dump_loc, "report.txt"), "w") as f:
        f.write(report)
    
    return val_result, test_result


def run_eval_with_demographics(
    eval_target: str,
    device: str,
    dump_loc: str,
    multiseed_run: bool=True,
    overwrite_params: dict=None
):
    report_manager = ReportManager(eval_target)

    if multiseed_run:
        dump_loc = os.path.join(dump_loc, "multirun", "eval")

    # Settings
    param_file = os.path.join(eval_target, "params.pkl")
    weightfile = os.path.join(eval_target, "net.pth")

    with open(param_file, "rb") as fp:
        params = pickle.load(fp)
    params.data_lim = None
    if overwrite_params is not None:
        params = vars(params)
        for key, val in overwrite_params.items():
            params[key] = val
        params = Namespace(**params)

    report_manager.add_row("Model", params.finetune_target)
    report_manager.add_row("Parameters", str(params))
    
    # Evaluator
    evaluator = ModelEvaluatorWithDemographics(
        params, dump_loc, device)
    evaluator.set_model()
    evaluator.set_lossfunc()
    evaluator.set_weight(weightfile)

    loader = evaluator.prepare_dataloader("test")
    test_result = evaluator.run(loader)

    report = report_manager.get_report()
    with open(os.path.join(dump_loc, "report.txt"), "w") as f:
        f.write(report)
    
    return test_result
