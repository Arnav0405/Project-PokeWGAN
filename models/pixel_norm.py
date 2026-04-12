import torch
import torch.nn as nn


class PixelNorm(nn.Module):
    """
    Pixel-wise feature vector normalization.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.eps)
