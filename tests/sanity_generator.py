import os
import sys
from pathlib import Path

import torch
from torchinfo import summary

# Ensure project root is on sys.path when this file is run directly, e.g.:
# python tests/sanity_generator.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.generator import Generator


def run_sanity_test(
    batch_size: int = 4,
    z_dim: int = 256,
    base_channels: int = 512,
    device: str = "cpu",
) -> None:
    print("Running Generator sanity test...")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, z_dim: {z_dim}, base_channels: {base_channels}")

    gen = Generator(z_dim=z_dim, base_channels=base_channels, out_channels=3).to(device)
    gen.eval()

    with torch.no_grad():
        noise = torch.randn(batch_size, z_dim, device=device)
        out = gen(noise)

    expected_shape = (batch_size, 3, 128, 128)

    print(f"Output shape: {tuple(out.shape)}")
    print(f"Output dtype: {out.dtype}")
    print(f"Output min/max: {out.min().item():.4f} / {out.max().item():.4f}")
    print(f"Any NaN: {torch.isnan(out).any().item()}")
    print(f"Any Inf: {torch.isinf(out).any().item()}")

    assert tuple(out.shape) == expected_shape, (
        f"Shape mismatch. Expected {expected_shape}, got {tuple(out.shape)}"
    )
    assert out.dtype == torch.float32, f"Expected float32 output, got {out.dtype}"
    assert not torch.isnan(out).any(), "Output contains NaN values."
    assert not torch.isinf(out).any(), "Output contains Inf values."
    assert out.min().item() >= -1.1 and out.max().item() <= 1.1, (
        "Output appears outside expected tanh range."
    )

    params = sum(p.numel() for p in gen.parameters())
    trainable_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print(f"Total params: {params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print("✅ Generator sanity test passed.")

    print("\nModel summary:")
    summary(gen, input_size=(batch_size, z_dim))


if __name__ == "__main__":
    run_sanity_test()
