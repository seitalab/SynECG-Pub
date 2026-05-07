import torch
import torch.nn as nn

from codes.models.ssl.mae import MaskedAutoencoder, ChunkEmbed
from codes.models.utils import get_1d_sincos_pos_embed

class Transformer(MaskedAutoencoder):

    def __init__(
        self, 
        Block: nn.Module,
        seqlen: int,
        chunk_size: int, 
        in_channels: int, 
        emb_dim: int,
        depth: int,
        num_heads: int, 
        mlp_ratio: float,
        norm_layer: nn.Module=nn.LayerNorm,
        norm_pix_loss: bool=False
    ):
        nn.Module.__init__(self)

        # Encoder
        self.chunk_embed = ChunkEmbed(seqlen, chunk_size, in_channels, emb_dim)
        num_chunks = self.chunk_embed.num_chunks

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_chunks+1, emb_dim),
            requires_grad=False
        )

        self.blocks = nn.ModuleList([
            Block(emb_dim, num_heads, mlp_ratio, qkv_bias=True, qk_norm=True, norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(emb_dim)

        # misc
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):

        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.chunk_embed.num_chunks),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.chunk_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def forward(self, x):
        # embed patches
        x = self.chunk_embed(x) # -> bs, num_chunk, emb_dim

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x
