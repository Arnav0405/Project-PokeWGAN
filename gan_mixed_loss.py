import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from models.discriminator import Discriminator
from models.generator import Generator


def compute_r1_penalty(
    real_images: torch.Tensor, discriminator: nn.Module
) -> torch.Tensor:
    """
    R1 regularisation: penalise the gradient of D w.r.t. real images.
    Returns the mean squared gradient norm over the batch.
    """
    real_images = real_images.requires_grad_(True)
    d_real = discriminator(real_images)  # (N, 1)

    # Sum over batch to get a scalar for autograd
    gradients = torch.autograd.grad(
        outputs=d_real.sum(),
        inputs=real_images,
        create_graph=True,
    )[0]

    # ||grad||^2 summed over all dims except batch, then mean over batch
    r1 = gradients.pow(2).flatten(1).sum(1).mean()
    return r1


def compute_r2_penalty(
    fake_images: torch.Tensor, discriminator: nn.Module
) -> torch.Tensor:
    """
    R2 regularisation: penalise the gradient of D w.r.t. generated images.
    Returns the mean squared gradient norm over the batch.
    """
    fake_images = fake_images.requires_grad_(True)
    d_fake = discriminator(fake_images)  # (N, 1)

    gradients = torch.autograd.grad(
        outputs=d_fake.sum(),
        inputs=fake_images,
        create_graph=True,
    )[0]

    r2 = gradients.pow(2).flatten(1).sum(1).mean()
    return r2


def discriminator_loss(
    discriminator: nn.Module,
    generator: nn.Module,
    real_images: torch.Tensor,
    noise: torch.Tensor,
    gamma: float = 10.0,
) -> torch.Tensor:
    """
    Mixed discriminator loss (Equation 2 in the original paper).

    L_D = -E[log D(x)]                     # real images (standard GAN)
          + E[softplus(D(G(z)))]           # non‑saturating fake term
          + (γ/2) * (R1 + R2) * D(G(z))    # gradient penalties
    """
    batch_size = real_images.size(0)

    # ----- Real‑image term -------------------------------------------------
    d_real = discriminator(real_images)  # (N, 1) probability
    loss_real = -torch.log(d_real + 1e-12).mean()  # -log(D(x))

    # ----- Fake‑image term -------------------------------------------------
    with torch.no_grad():
        fake_images_detached = generator(noise)  # no grads through G

    # Need gradients w.r.t. fake images for R2, so re‑require grad
    fake_images = fake_images_detached.detach().requires_grad_(True)
    d_fake = discriminator(fake_images)  # (N, 1) probability
    loss_fake = F.softplus(d_fake).mean()  # log(1‑exp(‑D))

    # ----- Gradient penalties ---------------------------------------------
    r1 = compute_r1_penalty(real_images, discriminator)
    r2 = compute_r2_penalty(fake_images, discriminator)

    # Weight the penalties by the average discriminator confidence on fakes
    d_fake_mean = d_fake.mean().detach()
    reg = (gamma / 2.0) * (r1 + r2) * d_fake_mean

    return loss_real + loss_fake + reg


def generator_loss(
    discriminator: nn.Module,
    generator: nn.Module,
    noise: torch.Tensor,
) -> torch.Tensor:
    """
    Non‑saturating generator loss (Equation 3).

    L_G = -E[log G(z)]
    """
    fake_images = generator(noise)
    d_fake = discriminator(fake_images)  # (N, 1) probability
    loss_g = -torch.log(d_fake + 1e-12).mean()
    return loss_g


def _dummy_real_batch(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Returns a batch of random “real” images in the expected range [-1, 1].
    In a real project you would replace this with a DataLoader over your dataset.
    """
    return torch.randn(batch_size, 3, 64, 64, device=device).clamp(0, 1)


def train_demo(
    epochs: int = 2,
    batch_size: int = 4,
    z_dim: int = 256,
    lr: float = 2e-4,
    gamma: float = 10.0,
    device: torch.device = torch.device("cpu"),
):
    # Initialise models
    generator = Generator(z_dim=z_dim).to(device)

    discriminator = Discriminator().to(device)

    z = torch.randn(4, 256)  # batch of 4 noise vectors
    loss = generator_loss(discriminator, generator, z)
    print(loss.item())
    # Optimisers
    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.99))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.99))

    for epoch in range(1, epochs + 1):
        # ----- Sample a batch ------------------------------------------------
        real_imgs = _dummy_real_batch(batch_size, device)
        noise = torch.randn(batch_size, z_dim, device=device)

        # ----- Update Discriminator -----------------------------------------
        opt_d.zero_grad()
        loss_d = discriminator_loss(
            discriminator=discriminator,
            generator=generator,
            real_images=real_imgs,
            noise=noise,
            gamma=gamma,
        )
        loss_d.backward()
        opt_d.step()

        # ----- Update Generator ---------------------------------------------
        opt_g.zero_grad()
        loss_g = generator_loss(
            discriminator=discriminator,
            generator=generator,
            noise=noise,
        )
        loss_g.backward()
        opt_g.step()

        print(
            f"Epoch {epoch:02d} | D loss: {loss_d.item():.4f} | G loss: {loss_g.item():.4f}"
        )


if __name__ == "__main__":
    # Run the tiny demo – it will print two lines (one per epoch)

    train_demo()
