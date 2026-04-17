import os
from typing import Any, Dict, Optional

import torch
from torch import autograd
from torch.cuda.amp import GradScaler, autocast


class WGANTrainer:
    """
    WGAN-GP trainer for existing Generator and Discriminator(Critic).

    Features:
      - Wasserstein loss + gradient penalty
      - critic_iterations critic updates before each generator update
      - CUDA + mixed precision training
      - logs: generator_loss, discriminator_loss, gp, grad_norm
      - checkpoint save/load helpers
    """

    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        z_dim: int,
        lr_g: float = 1e-4,
        lr_d: float = 1e-4,
        betas: tuple[float, float] = (0.0, 0.9),
        gp_lambda: float = 10.0,
        critic_iterations: int = 5,
        device: str = "cuda",
        use_amp: bool = True,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available. "
                "Set device='cpu' or use a CUDA-enabled environment."
            )

        self.device = torch.device(device)
        self.use_amp = bool(use_amp and self.device.type == "cuda")

        self.G = generator.to(self.device)
        self.D = discriminator.to(self.device)

        self.z_dim = z_dim
        self.gp_lambda = gp_lambda
        self.critic_iterations = critic_iterations

        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=lr_g, betas=betas)
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=lr_d, betas=betas)

        self.scaler_g = GradScaler(enabled=self.use_amp)
        self.scaler_d = GradScaler(enabled=self.use_amp)

        self.global_step = 0
        self.current_epoch = 0

    def _sample_noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.z_dim, device=self.device)

    def _gradient_penalty(
        self, real_imgs: torch.Tensor, fake_imgs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            gp: scalar tensor
            grad_norm_mean: scalar tensor
        """
        batch_size = real_imgs.size(0)

        eps = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolates = eps * real_imgs + (1.0 - eps) * fake_imgs
        interpolates.requires_grad_(True)

        # Keep GP in full precision for numerical stability.
        critic_interpolates = self.D(interpolates)

        grad_outputs = torch.ones_like(critic_interpolates, device=self.device)

        gradients = autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)  # ||∇x D(x_hat)||_2
        gp = self.gp_lambda * ((grad_norm - 1.0) ** 2).mean()

        return gp, grad_norm.mean()

    def train_step(self, real_imgs: torch.Tensor) -> Dict[str, float]:
        """
        One WGAN-GP step:
          - train critic critic_iterations times
          - then train generator once
        """
        self.G.train()
        self.D.train()

        real_imgs = real_imgs.to(self.device, non_blocking=True)
        batch_size = real_imgs.size(0)

        d_loss_value = 0.0
        gp_value = 0.0
        grad_norm_value = 0.0

        # -------------------------
        # 1) Critic updates first
        # -------------------------
        for _ in range(self.critic_iterations):
            self.opt_d.zero_grad(set_to_none=True)

            z = self._sample_noise(batch_size)
            with autocast(enabled=self.use_amp):
                fake_imgs = self.G(z)

                real_scores = self.D(real_imgs)
                fake_scores = self.D(fake_imgs.detach())

                wasserstein_d = fake_scores.mean() - real_scores.mean()

            gp, grad_norm = self._gradient_penalty(real_imgs, fake_imgs.detach())
            d_loss = wasserstein_d + gp

            self.scaler_d.scale(d_loss).backward()
            self.scaler_d.step(self.opt_d)
            self.scaler_d.update()

            d_loss_value = float(d_loss.detach().item())
            gp_value = float(gp.detach().item())
            grad_norm_value = float(grad_norm.detach().item())

        # -------------------------
        # 2) Generator update
        # -------------------------
        self.opt_g.zero_grad(set_to_none=True)

        z = self._sample_noise(batch_size)
        with autocast(enabled=self.use_amp):
            fake_imgs = self.G(z)
            fake_scores = self.D(fake_imgs)
            g_loss = -fake_scores.mean()

        self.scaler_g.scale(g_loss).backward()
        self.scaler_g.step(self.opt_g)
        self.scaler_g.update()

        g_loss_value = float(g_loss.detach().item())

        self.global_step += 1

        return {
            "generator_loss": g_loss_value,
            "discriminator_loss": d_loss_value,
            "gp": gp_value,
            "grad_norm": grad_norm_value,
        }

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        self.G.eval()
        z = self._sample_noise(n)
        return self.G(z)

    def save_checkpoint(
        self,
        checkpoint_path: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        state = {
            "generator": self.G.state_dict(),
            "discriminator": self.D.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "scaler_g": self.scaler_g.state_dict(),
            "scaler_d": self.scaler_d.state_dict(),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "z_dim": self.z_dim,
            "gp_lambda": self.gp_lambda,
            "critic_iterations": self.critic_iterations,
            "extra": extra or {},
        }
        torch.save(state, checkpoint_path)

    def load_checkpoint(
        self, checkpoint_path: str, strict: bool = True
    ) -> Dict[str, Any]:
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        self.G.load_state_dict(ckpt["generator"], strict=strict)
        self.D.load_state_dict(ckpt["discriminator"], strict=strict)
        self.opt_g.load_state_dict(ckpt["opt_g"])
        self.opt_d.load_state_dict(ckpt["opt_d"])
        self.scaler_g.load_state_dict(ckpt["scaler_g"])
        self.scaler_d.load_state_dict(ckpt["scaler_d"])

        self.global_step = int(ckpt.get("global_step", 0))
        self.current_epoch = int(ckpt.get("current_epoch", 0))

        return {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "z_dim": ckpt.get("z_dim"),
            "gp_lambda": ckpt.get("gp_lambda"),
            "critic_iterations": ckpt.get("critic_iterations"),
            "extra": ckpt.get("extra", {}),
        }
