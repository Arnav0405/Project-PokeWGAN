import torch
import torch.nn as nn

from .conv_block import ConvBlock
from .equalized_conv import EqualizedConv2d


class Discriminator(nn.Module):
    """
    Critic for fixed 64x64 RGB images (WGAN-compatible).

    Architecture:
      1) Equalized Conv (from RGB input)
      2) ConvBlock
      3) AvgPool2d(2)
      4) ConvBlock
      5) AvgPool2d(2)
      6) ConvBlock
      7) AvgPool2d(2)
      8) ConvBlock
      9) Final Equalized Conv to 1 channel
     10) Flatten output logits/scores (no Sigmoid)

    Input:
      - (N, 3, 64, 64)

    Output:
      - (N, 1) critic scores/logits (unbounded)
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        c1 = base_channels  # 64x64
        c2 = base_channels * 2  # 32x32
        c3 = base_channels * 4  # 16x16
        c4 = base_channels * 8  # 8x8

        self.from_rgb = EqualizedConv2d(
            in_channels, c1, kernel_size=3, stride=1, padding=1
        )
        self.from_rgb_act = nn.LeakyReLU(0.2, inplace=True)

        self.block1 = ConvBlock(c1, c1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.block2 = ConvBlock(c1, c2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.block3 = ConvBlock(c2, c3)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.block4 = ConvBlock(c3, c4)

        self.final_conv = EqualizedConv2d(c4, 1, kernel_size=8, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1:] != (3, 64, 64):
            raise ValueError(
                f"Expected input shape (N, 3, 64, 64), got {tuple(x.shape)}"
            )

        x = self.from_rgb_act(self.from_rgb(x))

        x = self.block1(x)
        x = self.pool1(x)  # 32x32

        x = self.block2(x)
        x = self.pool2(x)  # 16x16

        x = self.block3(x)
        x = self.pool3(x)  # 8x8

        x = self.block4(x)  # 8x8

        x = self.final_conv(x)  # (N, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (N, 1)
        return x
