from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

# 既存実装を利用
from codes.models.utils import get_1d_sincos_pos_embed
from codes.models.ssl.mae import ChunkEmbed


class ChannelLayerNorm1d(nn.Module):
    """
    Conv1d (B, C, L) 用の LayerNorm。
    C 次元に対して LayerNorm を適用する。
    """
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x = x.transpose(1, 2)      # -> (B, L, C)
        x = self.ln(x)
        x = x.transpose(1, 2)      # -> (B, C, L)
        return x

class TokenResBlock1d(nn.Module):
    """
    長さ保存 1D 残差ブロック (Conv1d x2, Norm, ReLU)。
    受容野拡大のため kernel_size / dilation を指定可能。
    入出力: (B, C, L) -> (B, C, L)
    """
    def __init__(
        self,
        channels: int,
        norm_layer: Callable[[int], nn.Module],
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size は長さ保存のため奇数にしてください"
        padding = ((kernel_size - 1) // 2) * dilation

        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1   = norm_layer(channels)
        self.relu  = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2   = norm_layer(channels)

        # He init
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

        # Norm 層の weight/bias 初期化（BN / GN / LN ラッパ等に対応）
        for bn in [self.bn1, self.bn2]:
            if hasattr(bn, "weight") and bn.weight is not None:
                nn.init.constant_(bn.weight, 1.0)
            if hasattr(bn, "bias") and bn.bias is not None:
                nn.init.constant_(bn.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out

class TokenResNet1d(nn.Module):
    """
    トークン軸上の 1D ResNet。
    - kernel_size / dilation で受容野を調整
    - use_dilation=True のとき、ブロックごとに dilation を 1,2,4,... と増やす
    """
    def __init__(
        self,
        channels: int,
        depth: int,
        norm_layer: Callable[[int], nn.Module],
        kernel_size: int = 3,
        use_dilation: bool = True,
        max_dilation: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        blocks = []
        for i in range(depth):
            if use_dilation:
                dilation = min(2 ** i, max_dilation)
            else:
                dilation = 1

            blocks.append(
                TokenResBlock1d(
                    channels=channels,
                    norm_layer=norm_layer,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        return self.blocks(x)

class MaskedAutoencoderResNet(nn.Module):
    def __init__(
        self,
        seqlen: Optional[int],
        chunk_size: int = 50,
        in_channels: int = 1,
        emb_dim: int = 256,
        enc_depth: int = 8,
        dec_dim: int = 128,
        dec_depth: int = 4,
        norm_layer_tokens: Callable[..., nn.Module] = nn.LayerNorm,
        cnn_norm_layer: Optional[Callable[[int], nn.Module]] = ChannelLayerNorm1d, 
        dropout: float = 0.0,
        norm_pix_loss: bool = False,
        strict_seq_len: bool = True,
        dynamic_seq_pad: bool = False,
        # 受容野拡大用のパラメタ
        enc_kernel_size: int = 3,
        dec_kernel_size: int = 3,
        enc_use_dilation: bool = True,
        dec_use_dilation: bool = True,
        enc_max_dilation: int = 8,
        dec_max_dilation: int = 8,
    ):
        super().__init__()

        # ---- Embed（チャンク化） ----
        self.chunk_embed = ChunkEmbed(
            seqlen=seqlen,
            chunk_size=chunk_size,
            in_chans=in_channels,
            embed_dim=emb_dim,
            norm_layer=None,
            bias=True,
            strict_seq_len=strict_seq_len,
            dynamic_seq_pad=dynamic_seq_pad,
        )
        num_chunks = self.chunk_embed.num_chunks  # L

        # ---- Positional / special tokens ----
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_chunks + 1, emb_dim), requires_grad=False)

        # ---- CNN 用 Norm のデフォルト: GroupNorm ----
        if cnn_norm_layer is None:
            def default_cnn_norm(num_channels: int) -> nn.Module:
                # num_channels を割り切る最大の {8,4,2,1} を使う
                for g in (8, 4, 2, 1):
                    if num_channels % g == 0:
                        return nn.GroupNorm(num_groups=g, num_channels=num_channels)
                # 念のため fallback
                return nn.GroupNorm(num_groups=1, num_channels=num_channels)
            cnn_norm_layer = default_cnn_norm

        # ---- Encoder (ResNet over token axis) ----
        self.encoder = TokenResNet1d(
            channels=emb_dim,
            depth=enc_depth,
            norm_layer=cnn_norm_layer,
            kernel_size=enc_kernel_size,
            use_dilation=enc_use_dilation,
            max_dilation=enc_max_dilation,
            dropout=dropout,
        )
        self.enc_norm = norm_layer_tokens(emb_dim)

        # ---- Decoder ----
        self.decoder_embed = nn.Linear(emb_dim, dec_dim, bias=True)
        self.mask_token    = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_chunks + 1, dec_dim), requires_grad=False)

        self.decoder = TokenResNet1d(
            channels=dec_dim,
            depth=dec_depth,
            norm_layer=cnn_norm_layer,
            kernel_size=dec_kernel_size,
            use_dilation=dec_use_dilation,
            max_dilation=dec_max_dilation,
            dropout=dropout,
        )
        self.decoder_norm = norm_layer_tokens(dec_dim)
        self.decoder_pred = nn.Linear(dec_dim, chunk_size * in_channels, bias=True)

        # ---- misc ----
        self.emb_dim = emb_dim
        self.dec_dim = dec_dim
        self.chunk_size = chunk_size
        self.in_channels = in_channels
        self.norm_pix_loss = norm_pix_loss

        # ---- init ----
        self._initialize_weights(get_1d_sincos_pos_embed)

    def _initialize_weights(self, get_1d_sincos_pos_embed: Callable):
        # 位置埋め込み (sin-cos, 非学習)
        # pos_embed: (1, L+1, D)
        pos = get_1d_sincos_pos_embed(self.emb_dim, int(self.chunk_embed.num_chunks), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        dec_pos = get_1d_sincos_pos_embed(self.dec_dim, int(self.chunk_embed.num_chunks), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(dec_pos).float().unsqueeze(0))

        # ChunkEmbed の Conv の初期化（Linear相当）
        w = self.chunk_embed.proj.weight.data  # (D, C_in, chunk_size)
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        # decoder/linear等
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    # ----- ユーティリティ（Transformer版と同じI/F） -----

    def chunkify(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        sequences: (N, C, S) -> (N, L, p*C)
        """
        p = self.chunk_embed.chunk_size
        assert sequences.shape[2] % p == 0
        c = sequences.shape[1]
        s = sequences.shape[2] // p
        x = sequences.reshape(shape=(sequences.shape[0], c, s, p))
        x = torch.einsum('ncsp->nspc', x)
        x = x.reshape(shape=(sequences.shape[0], s, p * c))
        return x

    def unchunkify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, L, p*C) -> sequences: (N, C, S)
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
    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        """
        x: (N, L, D)
        returns:
          x_masked: (N, L_keep, D)  # ★ L_keep トークンは時間順に並んでいる
          mask:     (N, L)          # 0=keep, 1=remove
          ids_restore: (N, L)
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        # ランダムノイズに従って keep / remove を決定
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)         # ランダムな順序
        ids_keep_unsorted = ids_shuffle[:, :len_keep]     # keep するインデックス集合
        ids_mask = ids_shuffle[:, len_keep:]              # mask するインデックス集合

        # ★ エンコーダには「時間順」で入れたいので sort する
        ids_keep, _ = torch.sort(ids_keep_unsorted, dim=1)  # (N, L_keep)

        # エンコーダ入力: keep したトークンのみ、時間順
        x_masked = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )  # (N, L_keep, D)

        # デコーダで元の順序に戻すための perm と ids_restore を計算
        # perm: [keep(時間順), mask(任意順序)] という新→元インデックスの対応
        perm = torch.cat([ids_keep, ids_mask], dim=1)   # (N, L)
        ids_restore = torch.argsort(perm, dim=1)        # 元→新

        # loss 用 mask (1=remove, 0=keep)
        mask = torch.ones([N, L], device=x.device)
        mask.scatter_(1, ids_keep, 0)

        return x_masked, mask, ids_restore

    # ----- Encoder / Decoder -----

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float = 0.75):
        """
        x: (B, C, S)
        returns: latent (B, 1+L_keep, D), mask (B, L), ids_restore (B, L)
        """
        # チャンク埋め込み
        x = self.chunk_embed(x)                             # (B, L, D)

        # 位置埋め込み（cls を除く）
        x = x + self.pos_embed[:, 1:, :]

        # ランダムマスキング
        x, mask, ids_restore = self.random_masking(x, mask_ratio)  # (B, L_keep, D)

        # cls token を付与
        cls_token = self.cls_token + self.pos_embed[:, :1, :]      # (1, 1, D)
        x = torch.cat([cls_token.expand(x.shape[0], -1, -1), x], dim=1)  # (B, 1+L_keep, D)

        # CNN（ResNet）へ: (B, D, L)
        x = x.transpose(1, 2)                           # -> (B, D, 1+L_keep)
        x = self.encoder(x)                             # 長さ保存
        x = x.transpose(1, 2)                           # -> (B, 1+L_keep, D)
        x = self.enc_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor):
        """
        x: (B, 1+L_keep, D=emb_dim) from encoder
        returns: pred (B, L, p*C)
        """
        # 埋め替え
        x = self.decoder_embed(x)                       # (B, 1+L_keep, dec_dim)

        # マスクトークンで埋めて元の長さに復元（cls を除いた長さLへ）
        B, T, C = x.shape     # T = 1+L_keep
        L = ids_restore.shape[1]
        mask_tokens = self.mask_token.repeat(B, L + 1 - T, 1)     # (B, L - L_keep, dec_dim)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)          # (B, L, dec_dim)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C))  # (B, L, dec_dim)
        x = torch.cat([x[:, :1, :], x_], dim=1)                    # (B, 1+L, dec_dim)

        # 位置埋め込み付与
        x = x + self.decoder_pos_embed

        # CNN デコーダ
        x = x.transpose(1, 2)                      # (B, dec_dim, 1+L)
        x = self.decoder(x)                        # 長さ保存
        x = x.transpose(1, 2)                      # (B, 1+L, dec_dim)
        x = self.decoder_norm(x)

        # 各トークンを元チャンクへ線形投影
        x = self.decoder_pred(x)                   # (B, 1+L, p*C)

        # cls を除外
        x = x[:, 1:, :]                            # (B, L, p*C)
        return x

    def forward_loss(self, sequences: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor):
        """
        sequences: (B, C, S)
        pred:      (B, L, p*C)
        mask:      (B, L) 0=keep, 1=remove
        """
        target = self.chunkify(sequences)           # (B, L, p*C)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var  = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()

        loss = (pred - target) ** 2                 # (B, L, p*C)
        loss = loss.mean(dim=-1)                    # (B, L)

        denom = mask.sum()
        if denom.item() > 0:
            loss = (loss * mask).sum() / denom      # マスク部のみ平均
        else:
            loss = loss.mean()                      # マスクが無い場合は全体平均

        return loss

    def forward(self, sequences: torch.Tensor, mask_ratio: float = 0.75):
        latent, mask, ids_restore = self.forward_encoder(sequences, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(sequences, pred, mask)
        return loss