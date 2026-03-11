from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


class SpectralConv2d(nn.Module):
    """
    2D Fourier spectral convolution layer used in Fourier Neural Operator (FNO).

    This layer performs:
        1. FFT on spatial dimensions
        2. Learnable complex linear transform on selected low-frequency modes
        3. Inverse FFT back to spatial domain

    Input shape:
        (batch_size, in_channels, height, width)

    Output shape:
        (batch_size, out_channels, height, width)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}.")
        if modes1 <= 0:
            raise ValueError(f"modes1 must be positive, got {modes1}.")
        if modes2 <= 0:
            raise ValueError(f"modes2 must be positive, got {modes2}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale: float = 1.0 / (in_channels * out_channels)

        self.weight_pos = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes1,
                modes2,
                dtype=torch.cfloat,
            )
        )
        self.weight_neg = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes1,
                modes2,
                dtype=torch.cfloat,
            )
        )

    @staticmethod
    def compl_mul2d(x: Tensor, weight: Tensor) -> Tensor:
        """
        Complex multiplication in Fourier space.

        Args:
            x:
                Shape (batch_size, in_channels, modes1, modes2), complex tensor.
            weight:
                Shape (in_channels, out_channels, modes1, modes2), complex tensor.

        Returns:
            Tensor of shape (batch_size, out_channels, modes1, modes2), complex tensor.
        """
        if not torch.is_complex(x):
            raise TypeError("x must be a complex tensor.")
        if not torch.is_complex(weight):
            raise TypeError("weight must be a complex tensor.")

        return torch.einsum("bihw,iohw->bohw", x, weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Real tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Real tensor of shape (batch_size, out_channels, height, width)
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected x to have shape (batch, channels, height, width), "
                f"but got shape {tuple(x.shape)}."
            )

        batch_size, in_channels, height, width = x.shape
        if in_channels != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels, "
                f"but got {in_channels}."
            )

        x_ft: Tensor = torch.fft.rfft2(x, dim=(-2, -1))

        modes1 = min(self.modes1, height)
        modes2 = min(self.modes2, width // 2 + 1)

        out_ft: Tensor = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, :modes1, :modes2] = self.compl_mul2d(
            x_ft[:, :, :modes1, :modes2],
            self.weight_pos[:, :, :modes1, :modes2],
        )
        out_ft[:, :, -modes1:, :modes2] = self.compl_mul2d(
            x_ft[:, :, -modes1:, :modes2],
            self.weight_neg[:, :, :modes1, :modes2],
        )

        x_out: Tensor = torch.fft.irfft2(out_ft, s=(height, width), dim=(-2, -1))
        return x_out

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"modes1={self.modes1}, "
            f"modes2={self.modes2}"
        )


class FNOBlock2d(nn.Module):
    """
    A single 2D Fourier Neural Operator block.

    The block contains:
        - one spectral convolution branch (global interaction in Fourier space)
        - one pointwise 1x1 convolution branch (local linear mixing in spatial space)
        - one activation function

    Computation:
        y = activation( spectral_conv(x) + pointwise_conv(x) )

    Input shape:
        (batch_size, channels, height, width)

    Output shape:
        (batch_size, channels, height, width)
    """

    def __init__(
        self,
        channels: int,
        modes1: int,
        modes2: int,
        activation: nn.Module | None = None,
        bias: bool = True,
    ) -> None:
        """
        Args:
            channels: Number of input/output channels.
            modes1: Number of retained Fourier modes along height dimension.
            modes2: Number of retained Fourier modes along width dimension.
            activation: Activation module. Defaults to nn.GELU().
            bias: Whether to use bias in the 1x1 convolution.
        """
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if modes1 <= 0:
            raise ValueError(f"modes1 must be positive, got {modes1}.")
        if modes2 <= 0:
            raise ValueError(f"modes2 must be positive, got {modes2}.")

        self.channels = channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.spectral_conv = SpectralConv2d(
            in_channels=channels,
            out_channels=channels,
            modes1=modes1,
            modes2=modes2,
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            bias=bias,
        )
        self.activation = activation if activation is not None else nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor of shape (batch_size, channels, height, width)
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected x to have shape (batch, channels, height, width), "
                f"but got shape {tuple(x.shape)}."
            )

        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected input with {self.channels} channels, "
                f"but got {x.shape[1]}."
            )

        x_spec: Tensor = self.spectral_conv(x)
        x_pointwise: Tensor = self.pointwise_conv(x)
        x_out: Tensor = x_spec + x_pointwise
        x_out = self.activation(x_out)
        return x_out

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, "
            f"modes1={self.modes1}, "
            f"modes2={self.modes2}, "
            f"activation={self.activation.__class__.__name__}"
        )


class FNO2d(nn.Module):
    """
    Full 2D Fourier Neural Operator network.

    Architecture:
        input -> lifting -> stacked FNOBlock2d -> projection -> output

    Expected input shape:
        (batch_size, in_channels, height, width)

    Expected output shape:
        (batch_size, out_channels, height, width)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        width: int,
        depth: int,
        lifting_channels: int | None = None,
        projection_channels: int | None = None,
        activation: nn.Module | None = None,
        block_activation_factory: type[nn.Module] | None = nn.GELU,
        pointwise_bias: bool = True,
    ) -> None:
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            modes1: Number of retained Fourier modes along height dimension.
            modes2: Number of retained Fourier modes along width dimension.
            width: Hidden channel width used throughout FNO blocks.
            depth: Number of stacked FNO blocks.
            lifting_channels: Intermediate channel size in the lifting head.
                If None, defaults to width.
            projection_channels: Intermediate channel size in the projection head.
                If None, defaults to 2 * width.
            activation: Activation used in lifting/projection heads.
                If None, defaults to nn.GELU().
            block_activation_factory: Factory used to create the activation
                module for each FNOBlock2d. Defaults to nn.GELU.
            pointwise_bias: Whether the pointwise 1x1 conv in each FNOBlock2d
                uses bias.
        """
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}.")
        if modes1 <= 0:
            raise ValueError(f"modes1 must be positive, got {modes1}.")
        if modes2 <= 0:
            raise ValueError(f"modes2 must be positive, got {modes2}.")
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}.")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.depth = depth
        self.lifting_channels = lifting_channels if lifting_channels is not None else width
        self.projection_channels = (
            projection_channels if projection_channels is not None else 2 * width
        )

        if self.lifting_channels <= 0:
            raise ValueError(
                f"lifting_channels must be positive, got {self.lifting_channels}."
            )
        if self.projection_channels <= 0:
            raise ValueError(
                "projection_channels must be positive, "
                f"got {self.projection_channels}."
            )

        self.head_activation = activation if activation is not None else nn.GELU()

        # Lifting: project raw input channels into hidden width.
        self.lifting = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.lifting_channels,
                kernel_size=1,
                bias=True,
            ),
            self._clone_activation(self.head_activation),
            nn.Conv2d(
                in_channels=self.lifting_channels,
                out_channels=width,
                kernel_size=1,
                bias=True,
            ),
        )

        # Stacked Fourier blocks.
        self.blocks = nn.ModuleList(
            [
                FNOBlock2d(
                    channels=width,
                    modes1=modes1,
                    modes2=modes2,
                    activation=(
                        block_activation_factory()
                        if block_activation_factory is not None
                        else nn.GELU()
                    ),
                    bias=pointwise_bias,
                )
                for _ in range(depth)
            ]
        )

        # Projection: map hidden representation to target output channels.
        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels=width,
                out_channels=self.projection_channels,
                kernel_size=1,
                bias=True,
            ),
            self._clone_activation(self.head_activation),
            nn.Conv2d(
                in_channels=self.projection_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=True,
            ),
        )

    @staticmethod
    def _clone_activation(module: nn.Module) -> nn.Module:
        """
        Create a fresh activation module of the same type.

        This avoids reusing the exact same module instance in multiple places.
        For common activations like GELU/ReLU/SiLU, this is typically safe and clean.
        """
        return type(module)()

    def forward_features(self, x: Tensor) -> Tensor:
        """
        Compute hidden features before final projection.

        Args:
            x: Tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Tensor of shape (batch_size, width, height, width)
        """
        self._validate_input(x)

        x = self.lifting(x)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Tensor of shape (batch_size, out_channels, height, width)
        """
        x = self.forward_features(x)
        x = self.projection(x)
        return x

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                f"Expected x to have shape (batch, channels, height, width), "
                f"but got shape {tuple(x.shape)}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels, "
                f"but got {x.shape[1]}."
            )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"modes1={self.modes1}, "
            f"modes2={self.modes2}, "
            f"width={self.width}, "
            f"depth={self.depth}, "
            f"lifting_channels={self.lifting_channels}, "
            f"projection_channels={self.projection_channels}"
        )


if __name__ == "__main__":
    model = FNO2d(
        in_channels=3,
        out_channels=1,
        modes1=16,
        modes2=16,
        width=64,
        depth=4,
    )

    x = torch.randn(2, 3, 64, 64)
    y = model(x)

    print("x.shape =", x.shape)
    print("y.shape =", y.shape)