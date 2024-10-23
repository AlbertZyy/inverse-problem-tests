
from typing import Tuple, Callable
from math import prod

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy


class DropedBCELoss(nn.Module):
    r"""Auto-droped binary cross-entropy loss."""
    value: Tensor

    def __init__(self, shape: Tuple[int, ...], momentum: float = 0.1, *, dtype=None, device=None):
        super().__init__()
        kwargs = {'dtype': dtype, 'device': device}
        self.shape = shape
        self.numel = prod(shape)
        self.momentum = momentum
        self.register_buffer("value", torch.zeros((self.numel,), **kwargs))

    def gate(self):
        scale = torch.max(self.value).detach_()
        return torch.rand_like(self.value) < self.value / scale # (numel, )

    def reset_running_stats(self):
        with torch.no_grad():
            self.value.zero_()

    def forward(self, input: Tensor, target: Tensor):
        if self.training:
            raw_bce = binary_cross_entropy(input, target, reduction='none') # (batch, *shape)
            raw_bce = raw_bce.reshape(raw_bce.shape[0], -1) # (batch, numel)
            self.value.lerp_(raw_bce.mean(0).detach_(), self.momentum) # (numel, )
            return raw_bce[:, self.gate()].mean()

        else:
            return binary_cross_entropy(input, target, reduction='mean')

    __call__: Callable[[Tensor, Tensor], Tensor]
