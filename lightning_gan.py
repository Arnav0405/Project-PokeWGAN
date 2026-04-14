import lightning as L
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from gan_mixed_loss import discriminator_loss, generator_loss
from models.discriminator import Discriminator
from models.generator import Generator


class GANLightningModule(L.LightningModule):
    def __init__(
        self,
        z_dim: int = 256,
        g_base_channels: int = 512,
        d_base_channels: int = 64,
        out_channels: int = 3,
        learning_rate: float = 0.001,
        beta: tuple[float, float] = (0.0, 0.99),
    ):
        """
        Lightning module for GAN training on fixed 64x64 images.

        Args:
            z_dim: Latent noise dimension for the generator.
            g_base_channels: Generator base channel width.
            d_base_channels: Discriminator base channel width.
            out_channels: Number of image channels (3 for RGB).
            learning_rate: Adam learning rate for both optimizers.
            beta: Adam betas tuple.
        """
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(
            z_dim=z_dim,
            base_channels=g_base_channels,
            out_channels=out_channels,
        )
        self.discriminator = Discriminator(
            in_channels=out_channels,
            base_channels=d_base_channels,
        )

        self.z_dim = z_dim
        self.learning_rate = learning_rate
        self.beta = beta

        # GAN training with two optimizers requires manual optimization in Lightning 2.x
        self.automatic_optimization = False

        # FID metric (expects uint8 images in [0, 255] with shape [N, 3, H, W])
        self.fid = FrechetInceptionDistance(feature=2048)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.learning_rate, betas=self.beta
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.learning_rate, betas=self.beta
        )
        return [opt_g, opt_d]

    def on_train_epoch_start(self):
        self.fid.reset()

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            real_images = batch[0]
        else:
            real_images = batch

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

        # Update FID with real and generated images (detach so metric tracking
        # does not interfere with gradients)
        with torch.no_grad():
            fake_images = self.generator(noise).detach()
            real_uint8 = ((real_images.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
            fake_uint8 = ((fake_images.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
            self.fid.update(real_uint8, real=True)
            self.fid.update(fake_uint8, real=False)

        # Log only epoch-level losses (no per-step logging)
        self.log(
            "discriminator_loss_epoch",
            d_loss.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "generator_loss_epoch",
            g_loss.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        return {
            "discriminator_loss_epoch": d_loss.detach(),
            "generator_loss_epoch": g_loss.detach(),
        }

    def on_train_epoch_end(self):
        # Log epoch first to keep CSV columns in desired order:
        # epoch, discriminator_loss_epoch, generator_loss_epoch, fid
        self.log(
            "epoch",
            float(self.current_epoch),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        fid_value = self.fid.compute()
        self.log(
            "fid",
            fid_value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
