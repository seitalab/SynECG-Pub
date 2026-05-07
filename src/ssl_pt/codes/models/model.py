from enum import Enum
from functools import partial
from argparse import Namespace
# from typing import Optional
# from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

# from codes.models.architectures.resnet import ResNet18, ResNet34, CNNdecoder
from codes.models.transformer import Transformer
from codes.models.ssl.mae import MaskedAutoencoder
from codes.models.ssl.rnn_mae import MaskedAutoencoderRNN
from codes.models.ssl.cnn_mae import MaskedAutoencoderCNN
from codes.models.ssl.mega_mae import MegaMaskedAutoencoder
from codes.models.ssl.luna_mae import LunaMaskedAutoencoder
from codes.models.ssl.resnet_mae import MaskedAutoencoderResNet
from codes.models.ssl.simclr import SimCLR
from codes.models.ssl.byol import BYOL
from codes.models.ssl.dino import DINO
from codes.models.ssl.ibot import iBOT, iBOT_CNN

class ArchName(Enum):

    TRANSFORMER = "transformer"
    TRANSFORMER_REGISTER = "transformer_register"
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    GRU = "gru"
    LSTM = "lstm"
    CNNMAE = "cnnmae"
    MEGA = "mega"
    LUNA = "luna"

class ModelName(Enum):

    MAE = "mae"
    SIMCLR = "simclr"
    BYOL = "byol"
    DINO = "dino"
    IBOT = "ibot"
    MOCO = "moco"

class ModelFactory:

    @staticmethod
    def create_clf_model(params: Namespace) -> nn.Module:
        backbone = ModelFactory.create_backbone(params)

        if params.clf_mode == "logistic_regression":
            if params.architecture in [ArchName.TRANSFORMER.value, ArchName.TRANSFORMER_REGISTER.value]:
                seqlen = int(params.max_duration * params.target_freq)
                backbone.set_embed_module(
                    ChunkEmbed(
                        seqlen,
                        params.chunk_len,
                        params.num_lead,
                        params.emb_dim
                ))
            return Classifier(backbone, params.emb_dim)
        else:
            return Predictor(
                params,
                backbone,
                ModelFactory.create_head(params),
            )

    @staticmethod
    def create_backbone(params: Namespace) -> nn.Module:
        if params.architecture in [ArchName.TRANSFORMER.value, ArchName.TRANSFORMER_REGISTER.value]:
            seqlen = int(params.max_duration*params.target_freq)

            return Transformer(
                Block,
                seqlen,
                chunk_size=params.chunk_len,
                in_channels=params.num_lead,
                emb_dim=params.emb_dim, 
                depth=params.depth, 
                num_heads=params.heads, 
                mlp_ratio=params.mlp_ratio,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )

        # elif params.architecture == ArchName.RESNET18.value:
        #     return ResNet18(params)
        # elif params.architecture == ArchName.RESNET34.value:
        #     return ResNet34(params)
        else:
            raise ValueError(f"Unknown architecture name: {params.architecture}")

    @staticmethod
    def create_ssl_model(params: Namespace) -> nn.Module:
        if params.ssl == ModelName.MAE.value:
            seqlen = int(params.max_duration*params.target_freq)
            if params.architecture in [ArchName.TRANSFORMER.value, ArchName.TRANSFORMER_REGISTER.value]:

                return MaskedAutoencoder(
                    Block=Block,
                    seqlen=seqlen, 
                    chunk_size=params.chunk_len,
                    in_channels=params.num_lead,
                    emb_dim=params.emb_dim, 
                    depth=params.depth, 
                    num_heads=params.heads,
                    decoder_emb_dim=params.dec_emb_dim, 
                    decoder_depth=params.dec_depth, 
                    decoder_num_heads=params.dec_heads,
                    mlp_ratio=params.mlp_ratio, 
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                )

            elif params.architecture in [ArchName.GRU.value, ArchName.LSTM.value]:

                return MaskedAutoencoderRNN(
                    seqlen=seqlen, 
                    chunk_size=params.chunk_len,
                    in_channels=params.num_lead,
                    emb_dim=params.emb_dim, 
                    depth=params.depth, 
                    decoder_emb_dim=params.dec_emb_dim, 
                    decoder_depth=params.dec_depth, 
                    rnn_type=params.architecture,
                    enc_bidirectional=params.enc_bidirectional,
                    dec_bidirectional=params.dec_bidirectional,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),                     
                )

            elif params.architecture in [ArchName.RESNET18.value]:

                return MaskedAutoencoderResNet(
                    seqlen=seqlen, 
                    chunk_size=params.chunk_len,
                    in_channels=params.num_lead,
                    emb_dim=params.emb_dim, 
                    enc_depth=params.depth, 
                    dec_dim=params.dec_emb_dim, 
                    dec_depth=params.dec_depth, 
                )

            elif params.architecture in [ArchName.MEGA.value]:

                return MegaMaskedAutoencoder(
                    seqlen=seqlen, 
                    chunk_size=params.chunk_len,
                    in_channels=params.num_lead,
                    emb_dim=params.emb_dim, 
                    depth=params.depth, 
                    num_heads=params.heads,
                    qkv_dim=params.emb_dim, # qkv_dim=emb_dim
                    ff_dim=int(params.emb_dim * params.mlp_ratio), # ff_dim=4*emb_dim
                    decoder_emb_dim=params.dec_emb_dim, 
                    decoder_depth=params.dec_depth, 
                    decoder_num_heads=params.dec_heads,
                    decoder_qkv_dim=params.dec_emb_dim,
                    decoder_ff_dim=int(params.dec_emb_dim * params.mlp_ratio),
                )

            elif params.architecture in [ArchName.LUNA.value]:

                return LunaMaskedAutoencoder(
                    seqlen=seqlen, 
                    chunk_size=params.chunk_len,
                    in_channels=params.num_lead,
                    emb_dim=params.emb_dim, 
                    qkv_dim=params.emb_dim, # qkv_dim=emb_dim
                    depth=params.depth, 
                    num_heads=params.heads,
                    num_pheads=params.pheads,
                    context_length=params.context_length,
                    decoder_emb_dim=params.dec_emb_dim, 
                    decoder_qkv_dim=params.dec_emb_dim,
                    decoder_depth=params.dec_depth, 
                    decoder_num_heads=params.dec_heads,
                    decoder_num_pheads=params.dec_pheads,
                    decoder_context_length=params.dec_context_length,
                    mlp_ratio=params.mlp_ratio,
                )

            elif params.architecture in [ArchName.CNNMAE.value]:

                return MaskedAutoencoderCNN(
                    seqlen=seqlen, 
                    chunk_size=params.chunk_len,
                    in_channels=params.num_lead,
                    emb_dim=params.emb_dim, 
                    depth=params.depth, 
                    decoder_emb_dim=params.dec_emb_dim, 
                    decoder_depth=params.dec_depth, 
                )

            else:
                raise NotImplementedError(f"Unknown architecture: {params.architecture}")

        elif params.ssl == ModelName.SIMCLR.value:
            backbone = ModelFactory.create_backbone(params)
            if params.architecture in [ArchName.TRANSFORMER.value, ArchName.TRANSFORMER_REGISTER.value]:
                seqlen = int(params.max_duration * params.target_freq)
                backbone_out_dim = params.emb_dim
                assert params.token_selection is not None and params.use_cls_token
            else:
                backbone_out_dim = params.backbone_out_dim
            
            return SimCLR(
                backbone, 
                backbone_out_dim,
                params.projection_dim,
                params.temperature,
                params.token_selection
            )

        elif params.ssl == ModelName.BYOL.value:
            backbone = ModelFactory.create_backbone(params)
            if params.architecture in [ArchName.TRANSFORMER.value, ArchName.TRANSFORMER_REGISTER.value]:
                seqlen = int(params.max_duration * params.target_freq)
                backbone_out_dim = params.emb_dim
                assert params.token_selection is not None and params.use_cls_token
            else:
                backbone_out_dim = params.backbone_out_dim
            
            return BYOL(
                backbone, 
                backbone_out_dim,
                params.projection_dim,
                params.hidden_dim,
                params.token_selection
            )

        elif params.ssl == ModelName.DINO.value:
            backbone = ModelFactory.create_backbone(params)
            if params.architecture in [ArchName.TRANSFORMER.value, ArchName.TRANSFORMER_REGISTER.value]:
                seqlen = int(params.max_duration * params.target_freq)
                backbone_out_dim = params.emb_dim
                assert params.token_selection is not None and params.use_cls_token
            else:
                backbone_out_dim = params.backbone_out_dim
            
            return DINO(
                backbone, 
                backbone_out_dim,
                params.projection_dim,
                params.hidden_dim,
                params.temperature_student,
                params.temperature_teacher,
                params.center_momentum,
                params.token_selection
            )            

        elif params.ssl == ModelName.IBOT.value:
            backbone = ModelFactory.create_backbone(params)
            if params.architecture in [ArchName.TRANSFORMER.value, ArchName.TRANSFORMER_REGISTER.value]:
                seqlen = int(params.max_duration * params.target_freq)
                backbone_out_dim = params.emb_dim
                assert params.token_selection is not None and params.use_cls_token
                return iBOT(
                    backbone, 
                    backbone_out_dim,
                    params.projection_dim,
                    params.hidden_dim,
                    params.ibot_mask_ratio,
                    params.temperature_student,
                    params.temperature_teacher,
                    params.center_cls_momentum,
                    params.center_patch_momentum,
                )                 
            else:
                return iBOT_CNN(
                    backbone, 
                    params.backbone_out_dim,
                    params.projection_dim,
                    params.hidden_dim,
                    params.ibot_mask_ratio,
                    params.temperature_student,
                    params.temperature_teacher,
                    params.center_cls_momentum,
                    params.center_patch_momentum,
                )
            
        else:
            raise ValueError(f"Unknown SSL model name: {params.ssl}")

    @staticmethod
    def create_head(params: Namespace) -> nn.Module:
        return HeadModule(
            params.backbone_out_dim, params.head_dim, params.n_head_layer)

class Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, emb_dim: int):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        # temp solution
        if h.dim() == 3:
            h = h[:, 0]
        return self.fc(h)

class Predictor(nn.Module):
    def __init__(
        self, 
        params,
        backbone: nn.Module,
        head: nn.Module
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.is_mae = params.ssl == ModelName.MAE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_mae:
            h, _, _ = self.backbone.forward_encoder(x, mask_ratio=0)
            h = self._select_features(h)
            h = self.fc(h)
        else:
            h = self.backbone(x)
        return self.head(h)

    def _select_features(self, h: torch.Tensor) -> torch.Tensor:
        if self.select_type == SelectType.CLS_TOKEN:
            return h[:, 0]
        elif self.select_type == SelectType.MEAN:
            return torch.mean(h, dim=1)
        else:
            raise ValueError(f"Unknown select type: {self.select_type}")

class HeadModule(nn.Module):
    def __init__(self, in_dim: int, head_dim: int, n_head_layer: int):
        super().__init__()
        self.layers = nn.ModuleList()
        if n_head_layer == 0:
            head_dim = in_dim
        else:
            self.layers.append(self._create_layer(in_dim, head_dim))        
            for _ in range(n_head_layer - 1):
                self.layers.append(self._create_layer(head_dim, head_dim))
        
        self.fc_final = nn.Linear(head_dim, 1)

    def _create_layer(self, in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.fc_final(x)

def prepare_model(params: Namespace, load_ssl: bool) -> nn.Module:
    if load_ssl:
        return ModelFactory.create_ssl_model(params)
    else:
        return ModelFactory.create_clf_model(params)
