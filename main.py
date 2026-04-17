from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.discriminator import Discriminator
from models.generator import Generator
from models.wgan_gp_trainer import WGANTrainer
from plotter import plot_images


def load_processed_dataset(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Processed tensor file not found: {path}")

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "images" not in payload:
        raise ValueError(
            f"Unexpected dataset format in {path}. Expected dict with key 'images'."
        )

    images = payload["images"]
    if not isinstance(images, torch.Tensor):
        raise TypeError("'images' entry is not a torch.Tensor")
    if images.dim() != 4 or images.shape[1:] != (3, 64, 64):
        raise ValueError(
            f"Expected image tensor shape [N, 3, 64, 64], got {tuple(images.shape)}"
        )

    return images


def denormalize_to_uint8(image_chw: torch.Tensor) -> torch.Tensor:
    """
    Convert generated CHW image tensor to uint8 [0, 255].

    Supports:
      - [0, 1] tensors  -> x * 255
      - [-1, 1] tensors -> ((x + 1) / 2) * 255
    """
    img = image_chw.detach().cpu().float()

    min_v = float(img.min().item())
    max_v = float(img.max().item())

    if min_v >= -1.0 and max_v <= 1.0 and min_v < 0.0:
        img = (img + 1.0) / 2.0

    img = img.clamp(0.0, 1.0)
    img = (img * 255.0).round().to(torch.uint8)
    return img


def plot_and_save_image_uint8(image_uint8_chw: torch.Tensor, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    image_hwc = image_uint8_chw.permute(1, 2, 0).numpy()

    plt.figure(figsize=(4, 4))
    plt.imshow(image_hwc)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/Generate WGAN-GP on Pokemon 64x64 tensor dataset"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a WGANTrainer checkpoint to resume from.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Load a checkpoint and generate images without training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path used in --generate-only mode.",
    )
    parser.add_argument(
        "--num-generate",
        type=int,
        default=64,
        help="Number of images to generate in --generate-only mode.",
    )
    return parser.parse_args()


def build_trainer(device: str) -> WGANTrainer:
    z_dim = 256
    g_base_channels = 512
    d_base_channels = 64

    lr = 5e-4
    betas = (0.9, 0.99)
    gp_lambda = 5.0
    critic_iterations = 8

    generator = Generator(z_dim=z_dim, base_channels=g_base_channels, out_channels=3)
    discriminator = Discriminator(in_channels=3, base_channels=d_base_channels)

    trainer = WGANTrainer(
        generator=generator,
        discriminator=discriminator,
        z_dim=z_dim,
        lr_g=lr,
        lr_d=lr,
        betas=betas,
        gp_lambda=gp_lambda,
        critic_iterations=critic_iterations,
        device=device,
        use_amp=True,
    )
    return trainer


def run_training(resume_checkpoint: str | None = None) -> None:
    data_path = Path("data/processed/pokemon_64_normalized.pt")
    batch_size = 64
    epochs = 750

    checkpoint_dir = Path("models/checkpoints")
    final_model_path = Path("models/trained/wgan_gp_final.pt")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script, but no GPU is available.")

    trainer = build_trainer(device="cuda")

    images = load_processed_dataset(data_path)
    dataset = TensorDataset(images)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    history: List[Dict[str, float]] = []
    start_epoch = 1

    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        ckpt_info = trainer.load_checkpoint(str(resume_path), strict=True)
        start_epoch = int(ckpt_info.get("current_epoch", 0)) + 1
        extra = ckpt_info.get("extra", {})
        if isinstance(extra, dict) and isinstance(extra.get("history"), list):
            history = extra["history"]

        print(
            f"Resumed from checkpoint: {resume_path} | "
            f"current_epoch={ckpt_info.get('current_epoch', 0)} | "
            f"global_step={ckpt_info.get('global_step', 0)} | "
            f"next_epoch={start_epoch}"
        )

    for epoch in range(start_epoch, epochs + 1):
        epoch_g = 0.0
        epoch_d = 0.0
        epoch_gp = 0.0
        epoch_grad_norm = 0.0
        steps = 0

        for (real_imgs,) in dataloader:
            metrics = trainer.train_step(real_imgs)
            epoch_g += metrics["generator_loss"]
            epoch_d += metrics["discriminator_loss"]
            epoch_gp += metrics["gp"]
            epoch_grad_norm += metrics["grad_norm"]
            steps += 1

        trainer.current_epoch = epoch

        epoch_metrics = {
            "epoch": float(epoch),
            "generator_loss": epoch_g / max(steps, 1),
            "discriminator_loss": epoch_d / max(steps, 1),
            "gp": epoch_gp / max(steps, 1),
            "grad_norm": epoch_grad_norm / max(steps, 1),
        }
        history.append(epoch_metrics)

        print(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"G: {epoch_metrics['generator_loss']:.6f} | "
            f"D: {epoch_metrics['discriminator_loss']:.6f} | "
            f"GP: {epoch_metrics['gp']:.6f} | "
            f"grad_norm: {epoch_metrics['grad_norm']:.6f}"
        )

        if epoch % 25 == 0:
            ckpt_path = checkpoint_dir / f"wgan_gp_epoch_{epoch:03d}.pt"
            trainer.save_checkpoint(
                str(ckpt_path),
                extra={"epoch_metrics": epoch_metrics, "history": history},
            )

    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(
        str(final_model_path),
        extra={"history": history, "epochs": epochs},
    )
    print(f"Saved final model checkpoint to: {final_model_path}")


@torch.no_grad()
def generate_and_plot_from_checkpoint(
    checkpoint_path: str, num_generate: int = 64
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for generation in this script, but no GPU is available."
        )

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    trainer = build_trainer(device="cuda")
    info = trainer.load_checkpoint(str(ckpt), strict=True)

    print(
        f"Loaded checkpoint: {ckpt} | "
        f"epoch={info.get('current_epoch', 'NA')} | "
        f"global_step={info.get('global_step', 'NA')}"
    )

    # Generate batch (N, 3, 64, 64), generator output is sigmoid => likely [0, 1]
    fake_batch = trainer.sample(num_generate).detach().cpu()

    # plot_images expects image tensors; keep normalized range expected by your plotter.
    # If your plotter assumes [0,1], uncomment next line:
    fake_batch = ((fake_batch + 1.0) / 2.0).clamp(0.0, 1.0).mul(255)
    print(fake_batch.size())

    plot_images(fake_batch)


if __name__ == "__main__":
    args = parse_args()

    if args.generate_only:
        if args.checkpoint is None:
            raise ValueError("Please provide --checkpoint when using --generate-only")
        generate_and_plot_from_checkpoint(
            checkpoint_path=args.checkpoint,
            num_generate=args.num_generate,
        )
    else:
        print("Else statement runnning")
        run_training(resume_checkpoint=args.resume)
