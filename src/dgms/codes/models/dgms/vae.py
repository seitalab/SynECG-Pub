import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.models.dgms.base import BaseDGM

class VAEBase(BaseDGM):

    def _update_beta(self, n_sample_passed):
        self.n_passed += n_sample_passed
        self.beta = min(
            self.beta + (self.final_beta - self.initial_beta) * self.n_passed / self.n_total,
            self.final_beta
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def calc_loss(self, recon_x, x, mu, logvar, mask):
        recon_loss = F.mse_loss(recon_x, x, reduction="none") # bs, num_lead, seqlen
        recon_loss = torch.sum(recon_loss * mask) / x.size(0)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + kl_div * self.beta

    def generate(self, z):
        return self.decoder(z)

class VariationalAutoEncoder(VAEBase):

    def __init__(
        self, 
        encoder, 
        decoder, 
        enc_out_dim, 
        z_dim,
        initial_beta,
        final_beta,
        n_total
    ):
        super(VariationalAutoEncoder, self).__init__()
        self.z_dim = z_dim
        self.encoder = encoder
        self.decoder = decoder

        self.enc_z_mean = nn.Linear(enc_out_dim, z_dim)
        self.enc_z_logvar = nn.Linear(enc_out_dim, z_dim)

        # For beta update.
        self.beta = initial_beta
        self.initial_beta = initial_beta
        self.n_total = n_total
        self.n_passed = 0
        self.final_beta = final_beta

    def forward(self, x, mask):
        """_summary_

        Args:
            x (Tensor): Tensor of shape (bs, n_lead=1, seqlen)
            mask (Tensor): Tensor of shape (bs, n_lead=1, seqlen)
                # 0: masked, 1: not masked
        Returns:
            loss: Loss value.
        """
        if mask is None:
            mask = torch.ones_like(x)

        enc_out = self.encoder(x * mask)
        z_mean = self.enc_z_mean(enc_out)
        z_sdev = self.enc_z_logvar(enc_out)
        
        z = self.reparameterize(z_mean, z_sdev)
        
        recon_x = self.decoder(z)
        # from codes.utils.utils import debug_draw_samples
        # debug_draw_samples(recon_x, "./tmp/recon_samples")
        # debug_draw_samples(x, "./tmp/src_samples")
        assert x.size() == recon_x.size()
        loss = self.calc_loss(recon_x, x, z_mean, z_sdev, mask)
        self._update_beta(x.size(0))
        return loss
    
# class VAE_check(nn.Module):
#     def __init__(self, enc_out_dim=256, z_dim=64):
#         super(VAE_check, self).__init__()
        
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(128, enc_out_dim, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#         )
        
#         self.fc_mu = nn.Linear(enc_out_dim * 312, z_dim)
#         self.fc_logvar = nn.Linear(enc_out_dim * 312, z_dim)
        
#         # Decoder
#         self.decoder_input = nn.Linear(z_dim, enc_out_dim * 312)
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose1d(enc_out_dim, 128, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
#             nn.Linear(4992, 5000)
#             # nn.Sigmoid(),
#         )
    
#     def encode(self, x):
#         h = self.encoder(x)
#         h = h.view(h.size(0), -1)
#         mu = self.fc_mu(h)
#         logvar = self.fc_logvar(h)
#         return mu, logvar
    
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
    
#     def generate(self, z):
#         h = self.decoder_input(z)
#         h = h.view(h.size(0), -1, 312)
#         x_recon = self.decoder(h)

#         return x_recon
    
#     def calc_loss(self, recon_x, x, mu, logvar):
#         BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
#         KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#         return BCE + KLD

#     def forward(self, x, mask):
#         if mask is None:
#             mask = torch.ones_like(x)

#         mu, logvar = self.encode(x * mask)
#         z = self.reparameterize(mu, logvar)
#         x_recon = self.generate(z)
#         # return x_recon, mu, logvar
#         loss = self.calc_loss(x_recon, x, mu, logvar)
#         return loss