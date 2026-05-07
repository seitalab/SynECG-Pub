import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D
from tslearn.metrics import SoftDTWLossPyTorch

# helper functions (from denoising_diffusion_pytorch)
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class GRULayer(nn.Module):
    def __init__(self, embed_dim, depth=1):
        super().__init__()
        self.gru = nn.GRU(
            embed_dim, embed_dim, num_layers=depth,
            batch_first=True, bidirectional=True
        )

    def forward(self, x):
        output, _ = self.gru(x)
        return output

class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Create a long enough 'pe' matrix of embeddings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return self.pe[:, :x.size(1)]

class ReversibleSequenceEmbedding(nn.Module):

    def __init__(self, n_chunks, chunk_len, embed_dim, emb_type):
        super().__init__()
        self.n_chunks = n_chunks
        self.embed_dim = embed_dim
        
        # Reversible position encoding
        self.pos_encode = SinusoidalPositionEncoding(
            n_chunks, chunk_len)
        
        if emb_type == 'v01':
            # Reversible dimension reduction
            self.encoder = nn.Linear(chunk_len, embed_dim)
            self.decoder = nn.Linear(embed_dim, chunk_len)
        elif emb_type == 'v02':
            # Reversible dimension reduction
            self.encoder = nn.Sequential(
                nn.Linear(chunk_len, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(embed_dim, chunk_len),
                nn.GELU(),
                nn.LayerNorm(chunk_len),
                nn.Linear(chunk_len, chunk_len)
            )
        elif emb_type == 'v03':
            self.encoder = nn.Sequential(
                nn.Linear(chunk_len, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                # Add GRU layer
                GRULayer(embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim*2),
                nn.Linear(embed_dim*2, embed_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(embed_dim, chunk_len),
                nn.GELU(),
                nn.LayerNorm(chunk_len),
                GRULayer(chunk_len),
                nn.GELU(),
                nn.LayerNorm(chunk_len*2),
                nn.Linear(chunk_len*2, chunk_len)
            )
        elif emb_type == 'v04':
            self.encoder = nn.Sequential(
                nn.Linear(chunk_len, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                # Add GRU layer
                GRULayer(embed_dim, depth=2),
                nn.GELU(),
                nn.LayerNorm(embed_dim*2),
                nn.Linear(embed_dim*2, embed_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(embed_dim, chunk_len),
                nn.GELU(),
                nn.LayerNorm(chunk_len),
                GRULayer(chunk_len, depth=2),
                nn.GELU(),
                nn.LayerNorm(chunk_len*2),
                nn.Linear(chunk_len*2, chunk_len)
            )
        elif emb_type == 'v05':
            self.encoder = nn.Sequential(
                nn.Linear(chunk_len, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                # Add GRU layer
                GRULayer(embed_dim, depth=4),
                nn.GELU(),
                nn.LayerNorm(embed_dim*2),
                nn.Linear(embed_dim*2, embed_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(embed_dim, chunk_len),
                nn.GELU(),
                nn.LayerNorm(chunk_len),
                GRULayer(chunk_len, depth=4),
                nn.GELU(),
                nn.LayerNorm(chunk_len*2),
                nn.Linear(chunk_len*2, chunk_len)
            )
        else:
            raise ValueError(f'Unknown embedding type: {emb_type}')

        # Reversible normalization
        self.norm_encoder = nn.LayerNorm(embed_dim)
        self.norm_decoder = nn.LayerNorm(embed_dim)

    def encode(self, x):
        # x shape: (batch_size, dim, seq_len)
        chunks = x.view(x.size(0), self.n_chunks, -1) # -> bs, n_chunks, chunk_size

        # Add positional encoding
        pos_encoded = chunks + self.pos_encode(chunks)
        
        # Reduce dimension
        encoded = self.encoder(chunks)
        # Normalize
        encoded = self.norm_encoder(encoded)
        
        return encoded

    def decode(self, z):
        # Denormalize: z.shape = bs, embed_dim, n_chunks
        decoded = self.norm_decoder(z.transpose(1, 2))#.transpose(1, 2)

        # Increase dimension
        decoded = self.decoder(decoded)#.transpose(1, 2)
        
        # Remove positional encoding (approximately)
        decoded = decoded - self.pos_encode(decoded)
        
        # Reshape back to original sequence
        decoded = decoded.reshape(decoded.size(0), 1, -1)
        
        return decoded

class DDPM(GaussianDiffusion1D):

    def prep_loss(self, loss_type: str):
        if loss_type == 'soft_dtw':
            self.loss_fn = SoftDTWLossPyTorch(gamma=0.1)
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        elif loss_type == "mse_sum":
            self.loss_fn = nn.MSELoss(reduction="sum")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        self.loss_type = loss_type

    def set_embed_module(
        self, 
        embed_dim,
        chunk_len,
        embed_type: str
    ):
        self.embed_module = ReversibleSequenceEmbedding(
            n_chunks = self.seq_length,
            chunk_len = chunk_len,
            embed_dim = embed_dim,
            emb_type = embed_type
        )

    def p_losses(self, x_start, t, noise = None):
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean(), model_out                    

    def forward(self, sequence, mask, *args, **kwargs):
        # sequence: (batch_size, dim, seq_len=5000)
        if mask is None:
            mask = torch.ones_like(sequence)
        sequence_masked = sequence * mask
        # sequence_encoded = self.encoder(sequence_masked)
        # sequence_encoded = sequence_encoded.unsqueeze(1) # -> bs, 1, n_chunks (=dim)
        sequence_encoded = self.embed_module.encode(sequence_masked) # -> bs, n_chunks, dim
        sequence_encoded = sequence_encoded.transpose(1, 2) # -> bs, dim, n_chunks (=seq_len)
        b, c, n, device, seq_length, = *sequence_encoded.shape, sequence.device, self.seq_length

        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        sequence_encoded = self.normalize(sequence_encoded)
        ddpm_loss, model_out = self.p_losses(sequence_encoded, t, *args, **kwargs)
        # model_out: bs, dim, n_chunks

        sequence_recon = self.embed_module.decode(model_out)
        recon_loss = self.loss_fn(sequence_recon, sequence)
        if self.loss_type == "mse_sum":
            recon_loss = recon_loss / sequence.size(-1) # divide by seq_len
        else:
            recon_loss = recon_loss.mean()
        
        # return ddpm_loss + self.lambda_recon * recon_loss
        return ddpm_loss, recon_loss

    def reconstruct(self, sequence, *args, **kwargs):
        sequence_encoded = self.embed_module.encode(sequence) # -> bs, n_chunks, dim
        sequence_encoded = sequence_encoded.transpose(1, 2) # -> bs, dim, n_chunks (=seq_len)
        b, c, n, device, seq_length, = *sequence_encoded.shape, sequence.device, self.seq_length

        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        sequence_encoded = self.normalize(sequence_encoded)
        ddpm_loss, model_out = self.p_losses(
            sequence_encoded, t, *args, **kwargs)
        # model_out: bs, dim, n_chunks

        sequence_recon = self.embed_module.decode(model_out)
        return sequence_recon
    
    def generate(self, z):
        model_out = self.sample(z.size(0))
        sequence_recon = self.embed_module.decode(model_out)
        # sequence_recon = self.decoder(z)
        return sequence_recon