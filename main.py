import logging
import random
from pathlib import Path

# Suppress PyTorch Triton warning:
# "torch.utils.flop_counter.py: triton not found; flop counting will not work for triton kernels"
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

import lightning as L
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from torchmetrics.image.fid import FrechetInceptionDistance

from data_cleaning.preprocessing import PokemonDataModule
from lightning_gan import GANLightningModule

manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)

# Fixed-resolution GAN setup
BATCH_SIZE = 64
EPOCHS = 200
IMAGE_SIZE = 64  # train GAN strictly at 64x64


def main():
    # Output directories
    checkpoint_dir = Path("models") / "checkpoints"
    final_model_dir = Path("models") / "trained"
    logs_dir = Path("models") / "logs"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed 64x64 tensors saved by preprocessing.py
    dm = PreprocessedPokemonDataModule(
        data_path="data/processed/pokemon_64_normalized.pt",
        batch_size=64,
        num_workers=4,
    )

    # Fixed 64x64 GAN config:
    # - Generator should output 64x64
    # - Discriminator should expect 64x64
    model = GANLightningModule(
        z_dim=256,
        g_base_channels=512,
        d_base_channels=64,
        out_channels=3,
        learning_rate=0.001,
        beta=(0.0, 0.99),
    )

    # Logger writes logs to disk continuously
    csv_logger = CSVLogger(save_dir=str(logs_dir), name="lightning_gan_64x64")

    # FID metric callback setup (lower is better)
    fid_metric = FrechetInceptionDistance(feature=2048)

    # Checkpoint callback based on lowest FID
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="gan64-epoch{epoch:03d}-fid{fid_score:.4f}",
        save_top_k=5,  # save top 5 checkpoints
        monitor="fid",
        mode="min",
        save_last=True,
    )

    # Trainer on GPU (fallback to CPU if unavailable)
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=csv_logger,
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10)],
        enable_progress_bar=True,
        log_every_n_steps=1,
    )

    # Attach FID metric to model for logging/checkpoint monitoring
    model.fid = fid_metric

    # Train
    trainer.fit(model, datamodule=dm)

    # Save final trained generator weights
    # final_model_path = final_model_dir / "gan_generator_64x64_final.pt"
    # torch.save(model.generator.state_dict(), final_model_path)

    # Generate and visualize fake images
    model.eval()
    device = model.device
    with torch.no_grad():
        z = torch.randn(1, model.z_dim, 1, 1, device=device)
        fake = model.generator(z).detach().cpu()

    img = (fake[0] + 1) / 2  # denormalize from [-1, 1] to [0, 1]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Generated 64x64 Pokemon")
    plt.show()


if __name__ == "__main__":
    main()
