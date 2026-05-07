import sys
import os
# from functools import partial
from argparse import Namespace

# import torch
import torch.nn as nn
# import torch.nn.functional as F
# from timm.models.vision_transformer import Block

_SSSD_STANDALONE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "sssd_standalone")
)
if _SSSD_STANDALONE_DIR not in sys.path:
    sys.path.insert(0, _SSSD_STANDALONE_DIR)
# from codes.models.transformer import Transformer
from sssd_standalone import SSSDECG

class ModelFactory:


    @staticmethod
    def create_gen_model(params: Namespace) -> nn.Module:
        if params.gen_method == "sssd":
            model = SSSDECG(config_path=params.cfg_file)
        else:
            raise ValueError(f"Unknown Generative model name: {params.gen}")
        return model

    # @staticmethod
    # def create_head(params: Namespace) -> nn.Module:
    #     return HeadModule(
    #         params.backbone_out_dim, params.head_dim, params.n_head_layer)

# class Classifier(nn.Module):
#     def __init__(self, backbone: nn.Module, emb_dim: int):
#         super().__init__()
#         self.backbone = backbone
#         self.fc = nn.Linear(emb_dim, 1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         h = self.backbone(x)
#         # temp solution
#         if h.dim() == 3:
#             h = h[:, 0]
#         return self.fc(h)

# class Predictor(nn.Module):
#     def __init__(
#         self, 
#         params,
#         backbone: nn.Module,
#         head: nn.Module
#     ):
#         super().__init__()
#         self.backbone = backbone
#         self.head = head
#         self.is_mae = params.ssl == ModelName.MAE

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.is_mae:
#             h, _, _ = self.backbone.forward_encoder(x, mask_ratio=0)
#             h = self._select_features(h)
#             h = self.fc(h)
#         else:
#             h = self.backbone(x)
#         return self.head(h)

#     def _select_features(self, h: torch.Tensor) -> torch.Tensor:
#         if self.select_type == SelectType.CLS_TOKEN:
#             return h[:, 0]
#         elif self.select_type == SelectType.MEAN:
#             return torch.mean(h, dim=1)
#         else:
#             raise ValueError(f"Unknown select type: {self.select_type}")

# class HeadModule(nn.Module):
#     def __init__(self, in_dim: int, head_dim: int, n_head_layer: int):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         if n_head_layer == 0:
#             head_dim = in_dim
#         else:
#             self.layers.append(self._create_layer(in_dim, head_dim))        
#             for _ in range(n_head_layer - 1):
#                 self.layers.append(self._create_layer(head_dim, head_dim))
        
#         self.fc_final = nn.Linear(head_dim, 1)

#     def _create_layer(self, in_dim: int, out_dim: int) -> nn.Sequential:
#         return nn.Sequential(
#             nn.Linear(in_dim, out_dim),
#             nn.ReLU(),
#             nn.Dropout(0.25),
#             nn.LayerNorm(out_dim)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         for layer in self.layers:
#             x = layer(x)
#         return self.fc_final(x)

def prepare_model(params: Namespace) -> nn.Module:
    return ModelFactory.create_gen_model(params)
