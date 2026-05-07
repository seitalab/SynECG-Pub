import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from functools import partial

# 既存ユーティリティ（ご提示のViT-MAE実装と同じ名前空間を想定）
from codes.models.utils import get_1d_sincos_pos_embed

class ChunkEmbed(nn.Module):
    """ 1D sequence to Chunk Embedding (提示コードを再掲) """
    def __init__(
        self,
        seqlen: Optional[int],
        chunk_size: int = 50,
        in_chans: int = 1,
        embed_dim: int = 256,
        norm_layer: Optional[Callable] = None,
        bias: bool = True,
        strict_seq_len: bool = True,
        dynamic_seq_pad: bool = False,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        if seqlen is not None:
            self.seqlen = seqlen
            self.grid_size = self.seqlen // self.chunk_size
            self.num_chunks = self.grid_size
        else:
            self.seqlen = None
            self.grid_size = None
            self.num_chunks = None

        self.strict_seq_len = strict_seq_len
        self.dynamic_seq_pad = dynamic_seq_pad

        self.proj = nn.Conv1d(
            in_chans, embed_dim, kernel_size=chunk_size, stride=chunk_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape

        if self.seqlen is not None:
            if self.strict_seq_len:
                assert L == self.seqlen, f"Input seqlen ({L}) doesn't match model ({self.seqlen})."
            elif not self.dynamic_seq_pad:
                assert L % self.chunk_size == 0, f"Input length ({L}) should be divisible by chunk size ({self.chunk_size})."
        if self.dynamic_seq_pad:
            pad = (self.chunk_size - L % self.chunk_size) % self.chunk_size
            x = F.pad(x, (0, pad))

        x = self.proj(x)         # -> (B, embed_dim, num_chunks)
        x = x.transpose(1, 2)    # -> (B, num_chunks, embed_dim)
        x = self.norm(x)
        return x

# ---- LayerNorm（チャネル次元）ヘルパ ----
class ChannelLayerNorm(nn.Module):
    """LayerNorm over channel dimension for tensors with shape [B, C, L]."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L] -> LN over C
        return self.ln(x.transpose(1, 2)).transpose(1, 2)

# ---- 1D Conv residual block（長さ保存）----
class ConvBlock1d(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, dilation: int = 1, drop: float = 0.0):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.norm1 = ChannelLayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=pad, dilation=dilation, bias=False)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(drop)
        self.norm2 = ChannelLayerNorm(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=pad, dilation=dilation, bias=False)

        # Kaiming init（ReLU/GELU系に適した初期化）
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        residual = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x + residual

# ---- CNN Encoder（トークン列をConvで処理）----
class ConvEncoder1D(nn.Module):
    def __init__(self, emb_dim: int, depth: int = 8, kernel_size: int = 3, drop: float = 0.0,
                 dilations: Optional[list] = None):
        super().__init__()
        if dilations is None:
            # 受容野を広げるために軽いダイレーションを付与
            base = [1, 1, 2, 2, 4, 4, 8, 8]
            dilations = base[:depth] if depth <= len(base) else [1] * depth

        self.blocks = nn.ModuleList([
            ConvBlock1d(emb_dim, kernel_size=kernel_size, dilation=d, drop=drop) for d in dilations
        ])
        self.norm = ChannelLayerNorm(emb_dim)

    def forward(self, x_tok: torch.Tensor) -> torch.Tensor:
        # x_tok: [B, L, D] -> [B, D, L]
        x = x_tok.transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # -> [B, L, D]
        return x.transpose(1, 2)

# ---- CNN Decoder（トークン→チャンクの元波形ベクトルへ）----
class ConvDecoder1D(nn.Module):
    def __init__(self, in_dim: int, dec_dim: int, out_dim: int, depth: int = 4, kernel_size: int = 3, drop: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, dec_dim, bias=True)

        self.blocks = nn.ModuleList([
            ConvBlock1d(dec_dim, kernel_size=kernel_size, dilation=1, drop=drop) for _ in range(depth)
        ])
        self.norm = ChannelLayerNorm(dec_dim)
        self.head = nn.Conv1d(dec_dim, out_dim, kernel_size=1, bias=True)

        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)
        nn.init.kaiming_normal_(self.head.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x: torch.Tensor, pos_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, L, in_dim]  pos_embed: [1, L, dec_dim] (optional, absolute pos)
        x = self.proj(x)  # [B, L, dec_dim]
        if pos_embed is not None:
            x = x + pos_embed

        # [B, L, C] -> [B, C, L] -> conv blocks -> norm -> head -> [B, out_dim, L] -> [B, L, out_dim]
        x = x.transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x)
        return x.transpose(1, 2)

# ---- CNN-MAE本体 ----
class MaskedAutoencoderCNN(nn.Module):
    """
    CNN-MAE for 1D signals (ECG).
    - Tokenization: ChunkEmbed (kernel=stride=chunk_size)
    - Encoder: length-preserving Conv residual stack over tokens
    - Decoder: Conv residual stack to predict chunk vectors (chunk_size * in_channels) per token
    - Loss: masked MSE on masked tokens only (同じ仕様)
    """
    def __init__(
        self,
        seqlen: int,
        chunk_size: int,
        in_channels: int,
        emb_dim: int,
        depth: int,
        decoder_emb_dim: int,
        decoder_depth: int,
        norm_pix_loss: bool = False,
        kernel_size: int = 3,
        drop: float = 0.0,
    ):
        super().__init__()

        # Tokenizer
        self.chunk_embed = ChunkEmbed(seqlen, chunk_size, in_channels, emb_dim)
        num_chunks = self.chunk_embed.num_chunks

        # Absolute pos. embeddings (encoder/decoderで共有 or 別)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_chunks, emb_dim), requires_grad=False)
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, num_chunks, decoder_emb_dim), requires_grad=False)

        # マスクトークン（Encoderで使用：Masked tokens are replaced with this vector）
        self.enc_mask_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        # Encoder/Decoder
        self.encoder = ConvEncoder1D(emb_dim=emb_dim, depth=depth, kernel_size=kernel_size, drop=drop)
        self.decoder = ConvDecoder1D(in_dim=emb_dim, dec_dim=decoder_emb_dim,
                                     out_dim=chunk_size * in_channels, depth=decoder_depth,
                                     kernel_size=kernel_size, drop=drop)

        # Misc
        self.norm_pix_loss = norm_pix_loss

        # init
        self.initialize_weights()

    # ==== utilities ====
    def initialize_weights(self):
        # sin-cos absolute pos
        pe = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.chunk_embed.num_chunks), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        dec_pe = get_1d_sincos_pos_embed(self.dec_pos_embed.shape[-1], int(self.chunk_embed.num_chunks), cls_token=False)
        self.dec_pos_embed.data.copy_(torch.from_numpy(dec_pe).float().unsqueeze(0))

        # mask token
        nn.init.normal_(self.enc_mask_token, std=0.02)

        # ChunkEmbed convの初期化（ViT-MAE実装に合わせてxavier）
        w = self.chunk_embed.proj.weight.data  # [emb_dim, in_chans, chunk_size]
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # 線形層/LayerNorm等の初期化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def chunkify(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        sequences: (N, C, S) -> target tokens: (N, L, p*C)
        """
        p = self.chunk_embed.chunk_size
        assert sequences.shape[2] % p == 0, "seqlen must be divisible by chunk_size"
        c = sequences.shape[1]
        s = sequences.shape[2] // p
        x = sequences.reshape(sequences.shape[0], c, s, p)  # (N,C,s,p)
        x = torch.einsum('ncsp->nspc', x)                  # (N,s,p,C)
        x = x.reshape(sequences.shape[0], s, p * c)        # (N,L,p*C)
        return x

    def unchunkify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, L, p*C) -> sequences: (N, C, S)
        """
        p = self.chunk_embed.chunk_size
        s = x.shape[1]                     # L
        c = x.shape[2] // p
        assert c * p == x.shape[2]
        x = x.reshape(x.shape[0], s, p, c) # (N,L,p,C)
        x = torch.einsum('nspc->ncsp', x)  # (N,C,s,p)
        sequences = x.reshape(x.shape[0], c, s * p)
        return sequences

    @torch.no_grad()
    def _random_mask(self, N: int, L: int, device, mask_ratio: float) -> torch.Tensor:
        """
        Returns binary mask [N, L]: 1=masked, 0=keep
        """
        len_keep = int(L * (1.0 - mask_ratio))
        noise = torch.rand(N, L, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)    # ascend
        mask = torch.ones(N, L, device=device)
        mask.scatter_(1, ids_shuffle[:, :len_keep], 0)
        return mask

    # ==== forward ====
    def forward(self, sequences: torch.Tensor, mask_ratio: float = 0.75):
        """
        sequences: [B, C, S]  (e.g., B x 12 x 5000 for 10s@500Hz)
        Returns: loss, pred_tokens, mask
         - pred_tokens: [B, L, p*C]
         - mask: [B, L] with 1 indicating masked tokens
        """
        # 1) tokenize
        x = self.chunk_embed(sequences)          # [B, L, D]
        B, L, D = x.shape

        # 2) random masking (binary mask only; encoderは全トークンを見る)
        mask = self._random_mask(B, L, x.device, mask_ratio=mask_ratio)  # [B, L]

        # 3) add absolute pos & replace masked tokens by learnable token
        pos = self.pos_embed                    # [1, L, D]
        x = x + pos
        mask_exp = mask.unsqueeze(-1)           # [B, L, 1]
        x = x * (1.0 - mask_exp) + (self.enc_mask_token + pos) * mask_exp

        # 4) CNN encoder (長さ保存)
        latent = self.encoder(x)                # [B, L, D]

        # 5) CNN decoder（posを再付与）
        pred = self.decoder(latent, pos_embed=self.dec_pos_embed)   # [B, L, p*C]

        # 6) masked MSE loss
        loss = self.forward_loss(sequences, pred, mask)             # scalar

        return loss#, pred, mask

    def forward_loss(self, sequences: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        sequences: [B, C, S]
        pred: [B, L, p*C]
        mask: [B, L] (1=masked)
        """
        target = self.chunkify(sequences)  # [B, L, p*C]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var  = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)              # [B, L], per-token MSE
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return loss

# ---- factory: 既存の mae_vit_base と同じparamsインターフェース ----
def mae_cnn_base(params):
    """
    期待する params の例:
      params.max_duration, params.freq, params.downsample
      params.chunk_len, params.num_lead
      params.emb_dim, params.depth
      params.dec_emb_dim, params.dec_depth
      （dropやkernel_sizeを持たせたい場合は拡張可）
    """
    seqlen = int(params.max_duration * params.freq / params.downsample)

    model = MaskedAutoencoderCNN(
        seqlen=seqlen,
        chunk_size=params.chunk_len,
        in_channels=params.num_lead,
        emb_dim=params.emb_dim,
        depth=params.depth,
        decoder_emb_dim=params.dec_emb_dim,
        decoder_depth=params.dec_depth,
        norm_pix_loss=getattr(params, "norm_pix_loss", False),
        kernel_size=getattr(params, "kernel_size", 3),
        drop=getattr(params, "dropout", 0.0),
    )
    return model
