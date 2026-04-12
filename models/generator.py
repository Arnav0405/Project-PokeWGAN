import torch
import torch.nn as nn

from .equalized_conv import EqualizedConv2d
from .equalized_transpose_conv import EqualizedConvTranspose2d
from .pixel_norm import PixelNorm
from .spatial_attn import SpatialAttention


class TransposeConvBlock(nn.Module):
    """
    1) Transpose Conv + LeakyReLU(0.2)
    2) PixelNorm
    3) Equalized Conv + LeakyReLU(0.2)
    4) PixelNorm
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.tconv = EqualizedConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.conv = EqualizedConv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.pn1 = PixelNorm()
        self.pn2 = PixelNorm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.tconv(x))
        x = self.pn1(x)
        x = self.act(self.conv(x))
        x = self.pn2(x)
        return x


class ResidualConvBlockV1(nn.Module):
    """
    Residual block:
      a = PN(Act(EqConv(x)))
      b = PN(Act(EqConv(a)))
      out = x + a + b
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = EqualizedConv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = EqualizedConv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.pn_a = PixelNorm()
        self.pn_b = PixelNorm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.pn_a(self.act(self.conv1(x)))
        b = self.pn_b(self.act(self.conv2(a)))
        return x + a + b


class ResidualConvBlockV2(nn.Module):
    """
    Residual block:
      a = PN(Act(EqConv(x)))
      b = PN(Act(EqConv(a)))
      s = EqConv(x)  # no activation
      out = a + b + s
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = EqualizedConv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = EqualizedConv2d(channels, channels, kernel_size=3, padding=1)
        self.skip = EqualizedConv2d(channels, channels, kernel_size=1, padding=0)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.pn_a = PixelNorm()
        self.pn_b = PixelNorm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.pn_a(self.act(self.conv1(x)))
        b = self.pn_b(self.act(self.conv2(a)))
        s = self.skip(x)
        return a + b + s


class AttentionConvBlock(nn.Module):
    """
    Attention residual block:
      a = PN(Act(SpatialAttention(EqConv(x))))
      b = PN(Act(EqConv(a)))
      s = EqConv(x)  # no activation
      out = a + b + s
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv_attn = EqualizedConv2d(channels, channels, kernel_size=3, padding=1)
        self.spatial_attn = SpatialAttention(kernel_size=7)
        self.conv2 = EqualizedConv2d(channels, channels, kernel_size=3, padding=1)
        self.skip = EqualizedConv2d(channels, channels, kernel_size=1, padding=0)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.pn_a = PixelNorm()
        self.pn_b = PixelNorm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.conv_attn(x)
        a = self.spatial_attn(a)
        a = self.pn_a(self.act(a))

        b = self.pn_b(self.act(self.conv2(a)))
        s = self.skip(x)
        return a + b + s


class Generator(nn.Module):
    """
    GAN Generator producing 128x128 RGB output from Gaussian noise.

    Expected input shape:
      - (N, z_dim) OR (N, z_dim, 1, 1)

    Architecture:
      - Stem projection to 4x4
      - 5 upscaling blocks (2x each):
          1) TransposeConvBlock
          2) ResidualConvBlockV1
          3) ResidualConvBlockV2
          4) AttentionConvBlock
          5) Equalized Conv block (with upsample + eq conv)
      - ToRGB Equalized Conv (3 channels) + Tanh
    """

    def __init__(
        self,
        z_dim: int = 256,
        base_channels: int = 512,
        out_channels: int = 3,
    ):
        super().__init__()
        self.z_dim = z_dim

        # Project z -> 4x4 feature map
        self.proj = EqualizedConvTranspose2d(
            z_dim, base_channels, kernel_size=4, stride=1, padding=0
        )
        self.proj_act = nn.LeakyReLU(0.2, inplace=True)
        self.proj_pn = PixelNorm()

        # Channel schedule across 5 upscales: 4->8->16->32->64->128
        c1 = base_channels  # 4x4
        c2 = base_channels // 2  # 8x8
        c3 = base_channels // 4  # 16x16
        c4 = base_channels // 8  # 32x32
        c5 = base_channels // 16  # 64x64
        c6 = max(base_channels // 32, 32)  # 128x128

        # 1) Transpose ConvBlock (upsample 2x)
        self.up1 = TransposeConvBlock(c1, c2)

        # 2) Residual ConvBlock + upsample
        self.up2_conv = EqualizedConvTranspose2d(
            c2, c3, kernel_size=4, stride=2, padding=1
        )
        self.up2_block = ResidualConvBlockV1(c3)

        # 3) Residual ConvBlock + upsample
        self.up3_conv = EqualizedConvTranspose2d(
            c3, c4, kernel_size=4, stride=2, padding=1
        )
        self.up3_block = ResidualConvBlockV2(c4)

        # 4) Attention ConvBlock + upsample
        self.up4_conv = EqualizedConvTranspose2d(
            c4, c5, kernel_size=4, stride=2, padding=1
        )
        self.up4_block = AttentionConvBlock(c5)

        # 5) Equalized Conv block + upsample
        self.up5_conv = EqualizedConvTranspose2d(
            c5, c6, kernel_size=4, stride=2, padding=1
        )
        self.up5_eq = EqualizedConv2d(c6, c6, kernel_size=3, padding=1)
        self.up5_act = nn.LeakyReLU(0.2, inplace=True)
        self.up5_pn = PixelNorm()

        # Final RGB conv
        self.to_rgb = EqualizedConv2d(c6, out_channels, kernel_size=1, padding=0)
        self.out_act = nn.Tanh()

    def _ensure_4d_noise(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            return z.unsqueeze(-1).unsqueeze(-1)
        if z.dim() == 4 and z.shape[2] == 1 and z.shape[3] == 1:
            return z
        raise ValueError(
            f"Noise must be shaped (N, z_dim) or (N, z_dim, 1, 1), got {tuple(z.shape)}"
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self._ensure_4d_noise(z)

        x = self.proj_pn(self.proj_act(self.proj(z)))  # 4x4

        x = self.up1(x)  # 8x8

        x = self.up2_conv(x)  # 16x16
        x = self.up2_block(x)

        x = self.up3_conv(x)  # 32x32
        x = self.up3_block(x)

        x = self.up4_conv(x)  # 64x64
        x = self.up4_block(x)

        x = self.up5_conv(x)  # 128x128
        x = self.up5_pn(self.up5_act(self.up5_eq(x)))

        x = self.to_rgb(x)
        x = self.out_act(x)
        return x
