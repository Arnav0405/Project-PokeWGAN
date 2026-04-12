import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EqualizedConv2d(nn.Module):
    """
    Equalized learning-rate Conv2d.
    Weight is scaled at runtime by sqrt(2 / fan_in).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = math.sqrt(2.0 / fan_in)
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.weight * self.scale,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )
