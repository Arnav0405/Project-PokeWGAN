import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def plot_images(real_batch):
    images = real_batch
    rows = real_batch.shape[0]
    grid = make_grid(images, padding=2, normalize=True, nrow=int(np.sqrt(rows)))

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
    plt.show()


def plot_and_save_losses(checkpoint_path: str, output_path: str) -> None:
    """
    Load a WGAN‑GP checkpoint that contains a ``history`` list and plot the
    training metrics (generator loss, discriminator loss, gradient penalty,
    and gradient norm) as curves. The figure is saved to ``output_path``.
    """
    # Load the checkpoint – it should contain a ``history`` key with a list of
    # dictionaries, each having ``epoch``, ``generator_loss``, ``discriminator_loss``,
    # ``gp`` and ``grad_norm`` entries.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "extra" not in checkpoint:
        raise KeyError("Checkpoint does not contain a 'extra' field.")
    if "history" not in checkpoint["extra"]:
        raise KeyError("Checkpoint does not contain a 'history' field in 'extra'.")

    history = checkpoint["extra"]["history"]
    num_epochs = int(checkpoint["extra"]["epoch_metrics"]["epoch"])
    epochs = list(range(1, num_epochs + 1))
    # print(epochs)
    if not isinstance(epochs, list):
        raise ValueError("The 'epochs' field in the checkpoint is not a list.")
    gen_losses = [item["generator_loss"] for item in history]
    disc_losses = [-1 * item["discriminator_loss"] for item in history]
    gps = [item["gp"] for item in history]
    grad_norms = [item["grad_norm"] for item in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, gen_losses, label="Generator Loss")
    plt.plot(epochs, disc_losses, label="Discriminator Loss")
    plt.plot(epochs, gps, label="Gradient Penalty")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Metric")
    plt.title("WGAN‑GP Training Metrics")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, grad_norms, label="Grad Norm")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Metric")
    plt.title("WGAN‑GP Training Metrics")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path.replace("metrics", "grad_norm"))
    plt.close()


if __name__ == "__main__":
    plot_and_save_losses(
        "./models/checkpoints/wgan_gp_epoch_750.pt", "./models/trained/metrics"
    )
