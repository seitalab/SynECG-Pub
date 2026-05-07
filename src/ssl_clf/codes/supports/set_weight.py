from glob import glob

import torch
from codes.models.prepare_model import (
    find_model_dir, 
    find_model_dir_extra, 
    find_model_dir_progress
)

def get_weight_file(pt_id: str, load_target: str="default") -> str:

    if load_target == "default":
        target_dir = find_model_dir(pt_id)
        tstamp = target_dir.split("/")[-1]

        # Avoid error in case like following (1 sec difference):
        # `pt0102/251117-150122/251117-150123-syn_ecg-minakuchi``
        # weight_file_loc = f"{target_dir}/{tstamp}-*"
        weight_file_loc = f"{target_dir}/{tstamp[:-1]}?-*"
        weight_file = glob(weight_file_loc + "/net.pth")[-1] # Load latest (should be only one)

    elif load_target == "extra":
        target_dir = find_model_dir_extra(pt_id)
        tstamp = target_dir.split("/")[-1]

        weight_file_loc = f"{target_dir}/{tstamp}-*"
        weight_file = glob(weight_file_loc + "/net.pth")[-1] # Load latest (should be only one)
    elif load_target == "progress":
        # pt_id: progress-ptXXXX-YYYE6
        target_base_dir = find_model_dir_progress(pt_id)
        tstamp = target_base_dir.split("/")[-1]

        # Avoid error in case like following (1 sec difference):
        # `pt0102/251117-150122/251117-150123-syn_ecg-minakuchi``
        weight_file_loc = f"{target_base_dir}/{tstamp[:-1]}?-*/interims"
        progress = pt_id.split("-")[2]
        net_file_name = f"/net_*{progress}*.pth"
        weight_file = glob(weight_file_loc + net_file_name)
        # if weight_file does not exist, pick the latest
        if len(weight_file) == 0:
            weight_file = sorted(glob(weight_file_loc + "/net_*.pth"))[-1]
        else:
            weight_file = weight_file[-1]
    else:
        raise NotImplementedError
    return weight_file


def set_weight(model, weight_file: str, ssl: str):

    if ssl == "mae":
        model = _set_weight_base(model, weight_file)
    elif ssl == "dino":
        model = _set_weight_base(model, weight_file)
    elif ssl == "ibot":
        del model.backbone.center
        del model.backbone.center_patch
        model = _set_weight_ibot(model, weight_file)
        del model.backbone.teacher
        del model.backbone.proj_teacher
    elif ssl == "byol":
        model = _set_weight_base(model, weight_file)
        del model.backbone.online_projector
        del model.backbone.target_projector
        del model.backbone.predictor
    elif ssl == "simclr":
        model = _set_weight_base(model, weight_file)
    else:
        raise NotImplementedError

    return model

def _set_weight_base(model, weight_file: str):
    """
    Args:

    Returns:
    
    """
    # 
    state_dict = dict(torch.load(weight_file, map_location="cpu")) # OrderedDict -> dict
        
    old_keys = list(state_dict.keys())
    for key in old_keys:
        new_key = key.replace("backbone.", "")

        if key.startswith("fc."):
            state_dict.pop(key)
            continue

        state_dict[new_key] = state_dict.pop(key)

    model.backbone.load_state_dict(state_dict)
    return model

def _set_weight_ibot(model, weight_file: str):

    state_dict = dict(torch.load(weight_file, map_location="cpu")) # OrderedDict -> dict

    old_keys = list(state_dict.keys())
    for key in old_keys:
        new_key = key.replace("backbone.", "")
        if key.startswith("center"):
            state_dict.pop(key)
            continue

        state_dict[new_key] = state_dict.pop(key)
    
    model.backbone.load_state_dict(state_dict)
    return model
