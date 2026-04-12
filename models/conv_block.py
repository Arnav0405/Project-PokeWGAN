import torch
import torch.nn as nn

from .equalized_conv import EqualizedConv2d


class ConvBlock(nn.Module):
    """
    Two-layer equalized-conv block with LeakyReLU(0.2) after each conv.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = EqualizedConv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1
        )
        self.conv2 = EqualizedConv2d(
            out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x
