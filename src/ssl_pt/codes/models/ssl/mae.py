from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.models.utils import get_1d_sincos_pos_embed

class ChunkEmbed(nn.Module):

    """ 1D sequence to Chunk Embedding
    from `https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py`
    """

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
        # super(ChunkEmbed, self).__init__()
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
                assert L % self.chunk_size == 0, f"Input height ({L}) should be divisible by patch size ({self.chunk_size})."
        if self.dynamic_seq_pad:
            pad = (self.chunk_size - L % self.chunk_size) % self.chunk_size
            x = F.pad(x, (0, pad))
        x = self.proj(x) # -> bs, emb_dim, num_chunk
        x = x.transpose(1, 2) # -> bs, num_chunk, emb_dim
        x = self.norm(x)
        return x

class MaskedAutoencoder(nn.Module):

    def __init__(
        self, 
        Block: nn.Module,
        seqlen: int,
        chunk_size: int, 
        in_channels: int, 
        emb_dim: int,
        depth: int,
        num_heads: int, 
        decoder_emb_dim: int,
        decoder_depth: int,
        decoder_num_heads: int, 
        mlp_ratio: float,
        norm_layer: nn.Module=nn.LayerNorm,
        norm_pix_loss: bool=False
    ):
        super().__init__()

        # Encoder
        self.chunk_embed = ChunkEmbed(
            seqlen, chunk_size, in_channels, emb_dim)
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

        # Decoder
        self.decoder_embed = nn.Linear(emb_dim, decoder_emb_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_emb_dim))
        
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_chunks+1, decoder_emb_dim),
            requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_emb_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_norm=True, norm_layer=norm_layer)
            for i in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_emb_dim)
        self.decoder_pred = nn.Linear(decoder_emb_dim, chunk_size*in_channels, bias=True)

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

        decoder_pos_embed = get_1d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], 
            int(self.chunk_embed.num_chunks),
            cls_token=True
        )
        self.decoder_pos_embed.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.chunk_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def chunkify(self, sequences):
        """
        sequences: (N, n_channels, seqlen)
        x: (N, L, chunk_size * n_channels)
        """
        p = self.chunk_embed.chunk_size
        assert sequences.shape[2] % p == 0

        c = sequences.shape[1] # number of channels
        s = sequences.shape[2] // p # number of chunks

        x = sequences.reshape(shape=(sequences.shape[0], c, s, p))
        x = torch.einsum('ncsp->nspc', x)
        x = x.reshape(shape=(sequences.shape[0], s, p * c))
        return x
    
    def unchunkify(self, x):
        """
        x: (N, L, chunk_size * n_channel)
        imgs: (N, n_channels, L)
        """
        p = self.chunk_embed.chunk_size
        s = x.shape[1] # number of chunks
        c = x.shape[2] // p # number of channels
        assert c * p == x.shape[2]
        
        x = x.reshape(shape=(x.shape[0], s, p, c))
        x = torch.einsum('nspc->ncsp', x)
        sequences = x.reshape(shape=(x.shape[0], c, s * p))
        return sequences
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.

        Args:
            x: [N, L, D], sequence
            mask_ratio (_type_): _description_
        """
        N, L, D = x.shape # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio=0.0):
        # embed patches
        x = self.chunk_embed(x) # -> bs, num_chunk, emb_dim

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, sequences, pred, mask):
        """
        sequences: [N, C, S]
        pred: [N, L, p*C]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.chunkify(sequences)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, sequences, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(sequences, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(sequences, pred, mask)
        return loss#, pred, mask

# def mae_vit_base(params):
#     seqlen = int(params.max_duration * params.freq / params.downsample)

#     model = MaskedAutoencoder(
#         Block=Block,
#         seqlen=seqlen, 
#         chunk_size=params.chunk_len,
#         in_channels=params.num_lead,
#         emb_dim=params.emb_dim, 
#         depth=params.depth, 
#         num_heads=params.heads,
#         decoder_emb_dim=params.dec_emb_dim, 
#         decoder_depth=params.dec_depth, 
#         decoder_num_heads=params.dec_heads,
#         mlp_ratio=params.mlp_ratio, 
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), 
#     )
#     return model
