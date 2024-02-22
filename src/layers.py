
from functools import reduce
from typing import Sequence, Optional

import torch
import torch.nn as nn

_dtype = torch.dtype
Tensor = torch.Tensor


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride=1, padding=0, dilation=1, bias=True, *,
                 dtype: Optional[_dtype]=None) -> None:
        super().__init__()
        self.depth_wise_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                                     stride=stride, padding=padding, dilation=dilation,
                                     groups=in_channels, bias=bias, dtype=dtype)
        self.point_wise_conv = nn.Conv2d(in_channels, out_channels, 1,
                                     padding=0, bias=False, dtype=dtype)

    def forward(self, input: Tensor) -> Tensor:
        return self.point_wise_conv(self.depth_wise_conv(input))


class ReparamConv(nn.Module):
    def __init__(self, in_channels: int, out_channls: int,
                 kernel_sizes: Sequence[int], momentum=0.1, *,
                 dtype: Optional[_dtype]=None) -> None:
        super().__init__()
        if len(kernel_sizes) <= 0:
            raise ValueError("empty kernel_sizes is not allowed.")

        self.conv = nn.ModuleList()

        for kernel_size in kernel_sizes:
            self.conv.append(
                nn.Sequential(
                    DepthwiseConv2d(in_channels, out_channls, kernel_size,
                              padding=kernel_size//2, bias=False, dtype=dtype),
                    nn.BatchNorm2d(out_channls, momentum=momentum, dtype=dtype),
                )
            )

        self.bn = nn.BatchNorm2d(out_channls, momentum=momentum, dtype=dtype)

    def forward(self, input: Tensor) -> Tensor:
        iterable = (conv(input) for conv in self.conv)
        out = reduce(torch.add, iterable)
        return self.bn(out)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, n_channels: int, ratio: float=1.0, *,
                 dtype: Optional[_dtype]=None) -> None:
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear_1 = nn.Linear(n_channels, int(n_channels*ratio), dtype=dtype)
        self.linear_2 = nn.Linear(int(n_channels*ratio), n_channels, dtype=dtype)

    def forward(self, input: Tensor) -> Tensor:
        assert input.ndim == 4
        BATCH, CHANNEL, _, _ = input.shape
        out = self.global_avg_pool(input.clone())
        out = out.view(BATCH, CHANNEL)
        out = torch.relu_(self.linear_1(out))
        out = torch.sigmoid_(self.linear_2(out))
        out = out.view(BATCH, CHANNEL, 1, 1)
        return input * out


class GRNUnit(nn.Module):
    """Global Response Normalization"""
    def __init__(self, *, eps: float=1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor) -> Tensor:
        assert input.ndim == 4
        BATCH, CHANNEL, _, _ = input.shape
        aggregated: Tensor = torch.linalg.matrix_norm(input, dim=(-2, -1))
        sum_agg = aggregated.sum(dim=1, keepdim=True)
        aggregated.div_(sum_agg + self.eps)
        return aggregated.view(BATCH, CHANNEL, 1, 1) * input
