import lightning as L
import torch

from gan_mixed_loss import discriminator_loss, generator_loss
from models.discriminator import Discriminator
from models.generator import Generator


class GANLightningModule(L.LightningModule):
    def __init__(
        self,
        z_dim=256,
        base_channels=512,
        out_channels=3,
        learning_rate=0.001,
        beta=(0.0, 0.99),
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(
            z_dim=z_dim, base_channels=base_channels, out_channels=out_channels
        )
        self.discriminator = Discriminator(base_channels=base_channels)

        self.z_dim = z_dim
        self.learning_rate = learning_rate
        self.beta = beta

        # GAN training with two optimizers requires manual optimization in Lightning 2.x
        self.automatic_optimization = False

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.learning_rate, betas=self.beta
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.learning_rate, betas=self.beta
        )
        return [opt_g, opt_d]

    def training_step(self, batch, batch_idx):
        real_images = batch  # Expects a tensor batch of real images
        noise = torch.randn(real_images.size(0), self.z_dim, device=self.device)

        opt_g, opt_d = self.optimizers()

        # 1) Update Discriminator
        opt_d.zero_grad()
        d_loss = discriminator_loss(
            discriminator=self.discriminator,
            generator=self.generator,
            real_images=real_images,
            noise=noise,
            gamma=10.0,
        )
        self.manual_backward(d_loss)
        opt_d.step()

        # 2) Update Generator
        opt_g.zero_grad()
        g_loss = generator_loss(
            discriminator=self.discriminator,
            generator=self.generator,
            noise=noise,
        )
        self.manual_backward(g_loss)
        opt_g.step()

        # Logs (written by logger, visible in progress bar)
        self.log(
            "discriminator_loss",
            d_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "generator_loss",
            g_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {
            "discriminator_loss": d_loss.detach(),
            "generator_loss": g_loss.detach(),
        }
