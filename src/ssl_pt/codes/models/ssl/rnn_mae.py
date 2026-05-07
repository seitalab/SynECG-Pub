from functools import partial
from typing import Callable, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class MaskedAutoencoderRNN(nn.Module):
    """
    Masked Autoencoder with GRU/LSTM for 1D sequences.

    入出力とマスク処理は ViT ベース実装に合わせ、エンコード／デコードのみ RNN に置換。
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
        rnn_type: Literal["gru", "lstm"] = "gru",
        enc_bidirectional: bool = False,
        dec_bidirectional: bool = False,
        rnn_dropout: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        norm_pix_loss: bool = False,
    ):
        super().__init__()

        # ========= Encoder (Chunk embed + RNN) =========
        self.chunk_embed = ChunkEmbed(seqlen, chunk_size, in_channels, emb_dim)
        num_chunks = self.chunk_embed.num_chunks

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_chunks + 1, emb_dim), requires_grad=False
        )

        self.rnn_type = rnn_type.lower()
        self.enc_bidirectional = enc_bidirectional
        self.dec_bidirectional = dec_bidirectional

        # Encoder RNN
        RNN = nn.GRU if self.rnn_type == "gru" else nn.LSTM
        self.encoder_rnn = RNN(
            input_size=emb_dim,
            hidden_size=emb_dim,
            num_layers=depth,
            batch_first=True,
            dropout=rnn_dropout if depth > 1 else 0.0,
            bidirectional=enc_bidirectional,
        )
        self.encoder_bi_proj = (
            nn.Linear(emb_dim * 2, emb_dim) if enc_bidirectional else nn.Identity()
        )
        self.encoder_norm = norm_layer(emb_dim)

        # ========= Decoder (RNN) =========
        self.decoder_embed = nn.Linear(emb_dim, decoder_emb_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_emb_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_chunks + 1, decoder_emb_dim), requires_grad=False
        )

        self.decoder_rnn = RNN(
            input_size=decoder_emb_dim,
            hidden_size=decoder_emb_dim,
            num_layers=decoder_depth,
            batch_first=True,
            dropout=rnn_dropout if decoder_depth > 1 else 0.0,
            bidirectional=dec_bidirectional,
        )
        self.decoder_bi_proj = (
            nn.Linear(decoder_emb_dim * 2, decoder_emb_dim) if dec_bidirectional else nn.Identity()
        )
        self.decoder_norm = norm_layer(decoder_emb_dim)
        self.decoder_pred = nn.Linear(decoder_emb_dim, chunk_size * in_channels, bias=True)

        # misc
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    # ---------- weight init ----------
    def initialize_weights(self):
        # pos emb (encoder)
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.chunk_embed.num_chunks),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # pos emb (decoder)
        decoder_pos_embed = get_1d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.chunk_embed.num_chunks),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize chunk_embed conv like Linear
        w = self.chunk_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # tokens
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        # linear / ln
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ---------- utilities ----------
    def chunkify(self, sequences):
        """
        sequences: (B, C, S)
        return:   (B, L, p*C)  # L = num_chunks, p = chunk_size
        """
        p = self.chunk_embed.chunk_size
        assert sequences.shape[2] % p == 0
        c = sequences.shape[1]
        s = sequences.shape[2] // p

        x = sequences.reshape(shape=(sequences.shape[0], c, s, p))
        x = torch.einsum('ncsp->nspc', x)
        x = x.reshape(shape=(sequences.shape[0], s, p * c))
        return x

    def unchunkify(self, x):
        """
        x: (B, L, p*C) -> (B, C, S)
        """
        p = self.chunk_embed.chunk_size
        s = x.shape[1]
        c = x.shape[2] // p
        assert c * p == x.shape[2]
        x = x.reshape(shape=(x.shape[0], s, p, c))
        x = torch.einsum('nspc->ncsp', x)
        sequences = x.reshape(shape=(x.shape[0], c, s * p))
        return sequences

    @torch.no_grad()
    def random_masking(self, x, mask_ratio):
        """
        x: (B, L, D)
        return:
            x_masked: (B, L_keep, D)  # keep 部分はランダム順（ノイズ昇順）
            mask:     (B, L)          # 0: keep, 1: remove（元の順序で）
            ids_restore: (B, L)       # 復元用インデックス（ViT版と同じ）
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)     # 昇順: 小さいほど keep
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    # ---------- encoder ----------
    def forward_encoder(self, x, mask_ratio=0.0):
        """
        入力 x: (B, C, S) -> chunk_embed -> (B, L, D)
        RNN は系列順に敏感なため、保持トークンのみ **時間順に並べ替えて** エンコードします。
        その後、デコーダ都合に合わせて **ランダム順に戻す** ことで ViT 版と同じ復元ロジックを保ちます。
        """
        # チャンク埋め込み
        x = self.chunk_embed(x)              # (B, L, D)
        x = x + self.pos_embed[:, 1:, :]     # 位置埋め込み（cls 除く）
        x_full = x                           # 後で並べ替えに使う

        # ランダムマスキング（保持トークンは「ノイズ昇順」≒ランダム順）
        x_kept_rand, mask, ids_restore = self.random_masking(x, mask_ratio)
        B, L, D = x.shape
        K = x_kept_rand.shape[1]             # len_keep

        # 時間順のインデックスを作る
        ids_shuffle = torch.argsort(ids_restore, dim=1)  # (= 元の ids_shuffle)
        ids_keep_rand = ids_shuffle[:, :K]                # ランダム順の保持位置（元のインデックス）
        ids_keep_time, noise_pos = torch.sort(ids_keep_rand, dim=1)  # 時間順の保持位置, その時の「ランダム順の位置」

        # 時間順に並べ替えた保持トークン列を作る
        x_kept_time = torch.gather(x_full, dim=1, index=ids_keep_time.unsqueeze(-1).expand(-1, -1, D))

        # cls 付与（pos 付き）
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x_time = torch.cat([cls_token.expand(B, -1, -1), x_kept_time], dim=1)  # (B, K+1, D)

        # RNN エンコード
        enc_out, _ = self.encoder_rnn(x_time)  # (B, K+1, D * num_directions)
        enc_out = self.encoder_bi_proj(enc_out)  # 双方向なら 2D -> D
        enc_out = self.encoder_norm(enc_out)

        # デコーダに合わせ、トークン部を「ランダム順」に戻す
        # noise_pos: (B, K) は「時間順の j 番目がランダム順で p 番目」に対応
        # 逆写像 (noise -> time idx) を作る
        idx_range = torch.arange(K, device=x.device).unsqueeze(0).expand(B, K)
        reorder_noise_to_time = torch.empty_like(noise_pos)
        reorder_noise_to_time.scatter_(1, noise_pos, idx_range)  # 位置 p に j を入れる

        tokens_time = enc_out[:, 1:, :]  # (B, K, D)
        tokens_rand = torch.gather(tokens_time, dim=1,
                                   index=reorder_noise_to_time.unsqueeze(-1).expand(-1, -1, D))
        out = torch.cat([enc_out[:, :1, :], tokens_rand], dim=1)  # (B, K+1, D) 先頭は cls

        return out, mask, ids_restore

    # ---------- decoder ----------
    def forward_decoder(self, x, ids_restore):
        """
        x: (B, K+1, D)
        ids_restore: (B, L)
        1) 次元合わせ
        2) マスクトークンを連結 (keep + mask)
        3) ids_restore で元の L 長に復元
        4) 位置埋め込みを足して RNN デコード
        """
        B, Kp1, D = x.shape

        # 次元合わせ
        x = self.decoder_embed(x)  # (B, K+1, Dd)
        Dd = x.shape[-1]

        # マスクトークン連結（注意：cls を除く K のみ）
        L = ids_restore.shape[1]
        K = Kp1 - 1
        mask_tokens = self.mask_token.repeat(B, L - K, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # (B, L, Dd)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, Dd))  # 元順

        # cls を先頭に戻す + 位置埋め込み
        x = torch.cat([x[:, :1, :], x_], dim=1)  # (B, L+1, Dd)
        x = x + self.decoder_pos_embed

        # RNN デコード
        dec_out, _ = self.decoder_rnn(x)          # (B, L+1, Dd * num_dir)
        dec_out = self.decoder_bi_proj(dec_out)   # 双方向なら 2Dd -> Dd
        dec_out = self.decoder_norm(dec_out)

        # 予測（cls 除く）
        pred = self.decoder_pred(dec_out)[:, 1:, :]  # (B, L, p*C)
        return pred

    # ---------- loss ----------
    def forward_loss(self, sequences, pred, mask):
        """
        sequences: (B, C, S)
        pred:      (B, L, p*C)
        mask:      (B, L)  0: keep, 1: remove
        """
        target = self.chunkify(sequences)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                # (B, L)
        loss = (loss * mask).sum() / mask.sum() # マスクされたチャンクのみ平均
        return loss

    # ---------- forward ----------
    def forward(self, sequences, mask_ratio=0.75):
        """
        sequences: (B, C, S)
        return: loss, pred, mask
        """
        latent, mask, ids_restore = self.forward_encoder(sequences, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)    # (B, L, p*C)
        loss = self.forward_loss(sequences, pred, mask)
        return loss#, pred, mask


# ===== factory =====
def mae_rnn_base(params, rnn_type: Literal["gru", "lstm"] = "gru"):
    """
    既存の mae_vit_base と同じ params を想定。
    heads / dec_heads / mlp_ratio は RNN 版では未使用ですが、そのまま params に残っていても問題ありません。
    追加で以下フィールドがあると反映されます（無い場合は既定値）:
      - enc_bidirectional: bool = False
      - dec_bidirectional: bool = False
      - rnn_dropout: float = 0.0
      - norm_pix_loss: bool = False
    """
    seqlen = int(params.max_duration * params.freq / params.downsample)
    enc_bi = getattr(params, "enc_bidirectional", False)
    dec_bi = getattr(params, "dec_bidirectional", False)
    rnn_dropout = getattr(params, "rnn_dropout", 0.0)
    norm_pix_loss = getattr(params, "norm_pix_loss", False)

    model = MaskedAutoencoderRNN(
        seqlen=seqlen,
        chunk_size=params.chunk_len,
        in_channels=params.num_lead,
        emb_dim=params.emb_dim,
        depth=params.depth,
        decoder_emb_dim=params.dec_emb_dim,
        decoder_depth=params.dec_depth,
        rnn_type=rnn_type,
        enc_bidirectional=enc_bi,
        dec_bidirectional=dec_bi,
        rnn_dropout=rnn_dropout,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss=norm_pix_loss,
    )
    return model
