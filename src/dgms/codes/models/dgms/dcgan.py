from typing import Dict

import torch
import torch.nn as nn

from codes.models.dgms.base import BaseDGM

class DCGAN(BaseDGM):

    def __init__(
        self, 
        generator, 
        discriminator, 
        z_dim, 
    ):
        super(DCGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.z_dim = z_dim

        self.criterion = nn.BCELoss()
    
    def forward(self, x, update_discriminator: bool):
        """
        Forward pass of the DCGAN.
        Args:
            X (torch.Tensor): Input tensor (real images)
        Returns:
            dict: Contains discriminator outputs for real and fake images,
                  and the generated fake images
        """
        batch_size = x.size(0)

        real_labels = torch.ones(batch_size).to(x.device)
        fake_labels = torch.zeros(batch_size).to(x.device)

        if update_discriminator:
            # Train Discriminator
        
            ## Process real images
            real_output = self.discriminator(x)
            d_loss_real = self.criterion(real_output, real_labels)

            ## Fake data
            # z = torch.randn(batch_size, self.z_dim).to(x.device)
            z = torch.rand(batch_size, self.z_dim).to(x.device)
            fake_data_d = self.generator(z) # Fake for discrimitor
            fake_output = self.discriminator(fake_data_d.detach())
            d_loss_fake = self.criterion(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
        else:
            d_loss = None

        # Train Generator
        # z = torch.randn(batch_size, self.z_dim).to(x.device)
        z = torch.rand(batch_size, self.z_dim).to(x.device)
        fake_data_g = self.generator(z)
        fake_output = self.discriminator(fake_data_g)
        g_loss = self.criterion(fake_output, real_labels)
        
        return d_loss, g_loss
    
    def generate(self, z):
        return self.generator(z)

# class DCGAN(BaseDGM):

#     def __init__(
#         self, 
#         generator, 
#         discriminator, 
#         z_dim, 
#     ):
#         self.generator = generator
#         self.discriminator = discriminator
#         self.z_dim = z_dim

#         self.criterion = nn.BCELoss()
    
#     def set_optimizer(self, lr, beta1):
#         self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
#         self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

#     def forward(self, X):
#         """
#         Forward pass of the DCGAN.
#         Args:
#             X (torch.Tensor): Input tensor (real images)
#         Returns:
#             dict: Contains discriminator outputs for real and fake images,
#                   and the generated fake images
#         """
#         batch_size = X.size(0)

#         # Process real images
#         real_output = self.discriminator(X)

#         # Generate fake images
#         noise = torch.randn(batch_size, self.z_dim, 1, 1, device=X.device)
#         fakes = self.generator(noise)
#         fake_output = self.discriminator(fakes)

#         loss = self.calc_loss(real_output, fake_output)
#         return loss

#     def generator_step(self, fake_output):
#         """Perform a training step for the generator"""
#         self.optimizer_G.zero_grad()
#         g_loss = -torch.mean(fake_output)
#         g_loss.backward()
#         self.optimizer_G.step()
#         return g_loss.item()

#     def discriminator_step(self, real_output, fake_output):
#         """Perform a training step for the discriminator"""
#         self.optimizer_D.zero_grad()
#         d_loss = -torch.mean(real_output) + torch.mean(fake_output)
#         d_loss.backward()
#         self.optimizer_D.step()
#         return d_loss.item()

#     def calc_loss(self, real_output, fake_output):

#         d_loss = self.model.discriminator_step(
#             real_output, fake_output.detach())
        
#         # Generator step
#         g_loss = self.model.generator_step(fake_output)
        
#         return d_loss, g_loss