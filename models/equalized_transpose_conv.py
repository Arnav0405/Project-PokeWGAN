import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EqualizedConvTranspose2d(nn.Module):
    """
    Equalized learning-rate ConvTranspose2d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        # ConvTranspose uses shape: (in_channels, out_channels, k, k)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = math.sqrt(2.0 / fan_in)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose2d(
            x,
            self.weight * self.scale,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
