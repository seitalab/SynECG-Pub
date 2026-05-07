import os
import sys
from argparse import Namespace
from glob import glob
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.append("../ssl_pt")
from codes.models.model import prepare_model
from codes.models.ssl.simclr import TokenSelector
from codes.models.model_12lead_expansion import prepare_model_12lead_expansion
cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    cfg = yaml.safe_load(f)

def find_model_dir(pt_id: int) -> str:
    """
    Args:
        pt_id (int):
    Returns:
        model_dir (str):
    """
    model_dir = os.path.join(
        cfg["experiment"]["path"]["save_root"],
        f"pt{pt_id//100:02d}s",
        f"pt{pt_id:04d}",
    )
    target_dir = sorted(glob(model_dir + "/??????-??????"))[-1] # Load latest
    return target_dir

def find_model_dir_extra(pt_id: str) -> str:
    """
    Args:
        pt_id (str):
    Returns:
        model_dir (str):
    """
    model_dir = cfg["ssl"]["eval_pt_model"]["fixed_setting"]["extra_pt_model"][pt_id]
    target_dir = sorted(glob(model_dir + "/??????-??????"))[-1] # Load latest
    return target_dir

def find_model_dir_progress(pt_id: str) -> str:
    """
    Args:
        pt_id (str):
    Returns:
        model_dir (str):
    """
    src_pt_id = pt_id.split("-")[1][2:] # progress-ptXXXX-YYYE6 -> XXXX
    base_model_dir = find_model_dir(int(src_pt_id))
    return base_model_dir # exp_config is the same as src_pt_id.

def find_model_dir_gru_and_resnet(pt_id: str) -> str:
    """
    Args:
        pt_id (str):
    Returns:
        model_dir (str):
    """
    src_pt_id = pt_id[2:6] # ptXXXXyyy -> ptXXXX
    base_model_dir = find_model_dir(int(src_pt_id))
    return base_model_dir

def prepare_pretrained_model(params: Namespace) -> nn.Module:
    """

    """
    # Load experiment params.
    if params.finetune_target.startswith("pt-extra"):
        target_dir = find_model_dir_extra(params.finetune_target)
        pt_id = target_dir.split("/")[-2]
        exp_cfg_file = os.path.join(target_dir, f"{pt_id}.yaml")
    elif params.finetune_target.startswith("progress-pt"):
        target_dir = find_model_dir_progress(params.finetune_target)
        exp_cfg_file = os.path.join(target_dir, "exp_config.yaml")
    elif params.finetune_target[6:] in ["gru", "resnet"]:
        target_dir = find_model_dir_gru_and_resnet(params.finetune_target)
        exp_cfg_file = os.path.join(target_dir, "exp_config.yaml")
    else:
        pt_id = int(params.finetune_target[2:]) # ptXXXX -> XXXX
        target_dir = find_model_dir(pt_id)
        exp_cfg_file = os.path.join(target_dir, "exp_config.yaml")

    # Load model params.
    with open(exp_cfg_file) as f:
        pt_model_params = yaml.safe_load(f)

    # temporal.
    if params.finetune_target.startswith("pt-extra"):
        pt_model_params = {k: v["param_val"] for k, v in pt_model_params.items()}
        pt_model_params["ssl"] = "mae"
        pt_model_params["architecture"] = "transformer"
        pt_model_params["target_freq"] = 500


    # Load model.
    pt_model_params = Namespace(**pt_model_params)
    model = prepare_model(pt_model_params, load_ssl=True)
    return model, pt_model_params

def prepare_clf_model(params: Namespace) -> nn.Module:
    """
    Args:
        params (Namespace):
    Returns:
        predictor (nn.Module):
        ssl (str): 
    """
    # Prepare backbone.
    if params.finetune_target is not None:
        if params.finetune_target.startswith("random-init"):
            is_random_init = True
            params.finetune_target = "pt0006" # Force random init -> temporal.
        else:
            is_random_init = False

        
        model_backbone, pt_model_params = prepare_pretrained_model(params)
        token_selection = \
            cfg["ssl"]["eval_pt_model"]["fixed_setting"]\
            ["ssl_params"][pt_model_params.ssl]["token_selection"]
        out_dim_key = \
            cfg["ssl"]["eval_pt_model"]["fixed_setting"]\
            ["ssl_params"][pt_model_params.ssl]["out_dim_key"]
        backbone_out_dim = vars(pt_model_params)[out_dim_key]
        ssl = pt_model_params.ssl
        foot = None

        if is_random_init:
            params.finetune_target = "random-init"

    else:
        ssl = None
        token_selection = None
        # if params.modelname == "transformer":
        print("Random init. Transformer backbone.")
        from common.model.transformer import Transformer
        backbone_out_dim = params.emb_dim
        foot = LinearEmbed(params)
        params.backbone_out_dim = params.clf_fc_dim
        params.feat_select = params.select_type
        model_backbone = Transformer(params)
        # params.finetune_target = "pt0006" # Force random init -> temporal.
        # model_backbone, pt_model_params = prepare_pretrained_model(params, force_random_init=True)
        # token_selection = \
        #     cfg["ssl"]["eval_pt_model"]["fixed_setting"]\
        #     ["ssl_params"][pt_model_params.ssl]["token_selection"]
        # out_dim_key = \
        #     cfg["ssl"]["eval_pt_model"]["fixed_setting"]\
        #     ["ssl_params"][pt_model_params.ssl]["out_dim_key"]
        # backbone_out_dim = vars(pt_model_params)[out_dim_key]
        # ssl = pt_model_params.ssl
        # foot = None        

        # elif params.modelname == "resnet18":
        #     from common.model.resnet import ResNet18
        #     foot = None
        #     emb_dim = None
        #     params.backbone_out_dim = params.clf_fc_dim
        #     model_backbone = ResNet18(params)    

        # else:
        #     raise
    
    if getattr(params, "all_lead_expansion", False):
        model = prepare_model_12lead_expansion(
            params, model_backbone
        )

    # Prepare model.
    if params.clf_mode == "logistic_regression":
        model = Classifier(model_backbone, backbone_out_dim)
    elif params.clf_mode == "dnn":
        head = HeadModule(backbone_out_dim)
        model = Predictor(
            model_backbone, 
            head, 
            is_ssl=ssl is not None,
            token_selection=token_selection,
            foot=foot
        )
    else:
        raise
    return model, ssl

class Classifier(nn.Module):

    def __init__(self, mae, emb_dim):
        super(Classifier, self).__init__()

        self.mae = mae
        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, x):

        h, _, _ = self.mae.forward_encoder(x, mask_ratio=0) # (bs, num_chunks, emb_dim)
        h = self.fc(h[:, 0]) # use cls_token.
        return h

class Predictor(TokenSelector):

    def __init__(
        self, 
        backbone: nn.Module, 
        head: nn.Module,
        is_ssl: bool,
        token_selection: str="cls",
        foot: Optional[nn.Module]=None
    ) -> None:
        super(Predictor, self).__init__()

        self.backbone = backbone
        self.head = head

        self.is_ssl = is_ssl
        self.token_selection = token_selection
        self.foot = foot

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Tensor of size 
                (batch_size, num_lead, seq_len).
        Returns:
            h (torch.Tensor): Tensor of size (batch_size, num_classes)
        """
        if self.is_ssl:
            h = self.backbone.forward_encoder(x)
            if type(h) is tuple:
                h = h[0] # MAE, iBoT returns tuple.
            h = self._select_token(h)
        else:
            if self.foot is not None:
                x = self.foot(x)
            h = self.backbone(x)

        # Add head.        
        h = self.head(h)
        return h

class HeadModule(nn.Module):

    def __init__(self, in_dim: int):
        super(HeadModule, self).__init__()

        self.fc1 = nn.Linear(in_dim, 32)
        self.drop1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Tensor of size (num_batch, in_dim).
        Returns:
            feat (torch.Tensor): Tensor of size (num_batch, 1).
        """
        feat = F.relu(self.fc1(x))
        feat = self.drop1(feat)
        feat = self.fc2(feat)
        return feat