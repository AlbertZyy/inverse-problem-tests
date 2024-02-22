

import torch
import torch.nn as nn

from nngal.modules import DepthwiseConv2d, ReparamConv2d, SEBlock2d, GRNUnit

Tensor = torch.Tensor


class LarKConvBlock(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.lark_conv = ReparamConv2d(
            num_channels, num_channels,
            [9, 7, 5, 3],
            momentum=0.1, eps=1e-5, dtype=torch.float32
        )
        self.se_block = SEBlock2d(num_channels, 0.25, dtype=torch.float32)
        self.grn_unit = GRNUnit(eps=1e-5)

    def forward(self, input: Tensor) -> Tensor:
        output = self.lark_conv(input)
        output = self.se_block(output)
        output = self.grn_unit(output)
        return output
