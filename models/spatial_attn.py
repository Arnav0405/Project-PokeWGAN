import torch
import torch.nn as nn

from .equalized_conv import EqualizedConv2d


class SpatialAttention(nn.Module):
    """
    Simple spatial attention:
      attn = sigmoid(conv([avg_pool_c(x), max_pool_c(x)]))
      output = x * attn
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = EqualizedConv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn
