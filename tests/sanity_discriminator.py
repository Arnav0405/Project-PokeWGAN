import os
import sys

import torch

# Ensure project root is on PYTHONPATH when running this file directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import Discriminator


def run_sanity_discriminator() -> None:
    """
    Simple sanity checks for Discriminator:
    1) Forward pass shape check
    2) Probability range check after sigmoid
    3) Input shape validation check
    """
    torch.manual_seed(42)

    batch_size = 8
    img_size = 128
    x = torch.randn(batch_size, 3, img_size, img_size)

    model = Discriminator(in_channels=3, base_channels=img_size)
    model.eval()

    with torch.no_grad():
        y = model(x)

    # 1) Output shape
    assert y.shape == (batch_size, 1), (
        f"Expected output shape {(batch_size, 1)}, got {tuple(y.shape)}"
    )

    # 2) Sigmoid probability range
    assert torch.all(y >= 0.0) and torch.all(y <= 1.0), (
        "Discriminator output is not in [0, 1]"
    )

    print("✅ Discriminator forward pass sanity checks passed.")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Min prob: {y.min().item():.6f}, Max prob: {y.max().item():.6f}")

    # 3) Bad input shape should raise ValueError
    bad_x = torch.randn(batch_size, 3, 64, 64)
    try:
        _ = model(bad_x)
        raise AssertionError(
            "Expected ValueError for wrong input shape, but no error was raised."
        )
    except ValueError as exc:
        print("✅ Input-shape validation check passed.")
        print(f"Raised error: {exc}")


if __name__ == "__main__":
    run_sanity_discriminator()
