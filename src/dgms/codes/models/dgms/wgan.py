import torch

from codes.models.dgms.base import BaseDGM

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1).to(real_samples.device)
    interpolates = (
        alpha * real_samples + \
            ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
    d_interpolates = D(interpolates)

    fake = torch.ones(real_samples.size(0)).to(real_samples.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class WGAN_GP(BaseDGM):

    def __init__(
        self, 
        generator, 
        discriminator, 
        z_dim, 
        lambda_gp
    ):
        super(WGAN_GP, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.z_dim = z_dim
        self.lambda_gp = lambda_gp

    def forward(self, x, update_discriminator: bool):
        """
        Forward pass of the WGAN.
        Args:
            X (torch.Tensor): Input tensor (real images)
        Returns:
            dict: Contains discriminator outputs for real and fake images,
                  and the generated fake images
        """
        batch_size = x.size(0)

        if update_discriminator:
            # Train Discriminator
        
            ## Process real images
            real_output = self.discriminator(x)

            ## Fake data
            z = torch.randn(batch_size, self.z_dim).to(x.device)
            fake_data_d = self.generator(z).detach() # Fake for discrimitor
            fake_output = self.discriminator(fake_data_d)

            # Calc. gradient penalty
            if self.training:
                gradient_penalty = compute_gradient_penalty(
                    self.discriminator, x, fake_data_d)
            else:
                gradient_penalty = 1 / self.lambda_gp # to make it to 1.

            # Calc. discriminator loss
            d_loss = -torch.mean(real_output) +\
                torch.mean(fake_output) +\
                self.lambda_gp * gradient_penalty
        else:
            d_loss = None

        # Train Generator
        z = torch.randn(batch_size, self.z_dim).to(x.device)
        fake_data_g = self.generator(z)
        fake_output = self.discriminator(fake_data_g)
        g_loss = -torch.mean(fake_output)
        
        return d_loss, g_loss
    
    def generate(self, z):
        return self.generator(z)