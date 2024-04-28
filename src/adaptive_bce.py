
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F


class AdaptiveBCELoss(Module):
    running_prop: Tensor

    def __init__(self, num_channels: int, width: int, height: int,
                 momentum: float=0.99,
                 order: float=1.0, *,
                 eps: float=1e-8,
                 dtype=None, device=None) -> None:
        super().__init__()
        kwargs = dict(dtype=dtype, device=device)
        self.num_channels = num_channels
        self.width = width
        self.height = height
        self.momentum = momentum
        self.order = order
        self.eps = eps
        shape = (num_channels, width, height)
        initial = 1/width/height
        self.register_buffer('running_prop', torch.full(shape, initial, **kwargs))

    # shape: (batch, c, h, w)
    def forward(self, input: Tensor, target: Tensor):
        raw = F.binary_cross_entropy(input, target, reduction='none').mean(dim=0)
        if self.training:
            assert raw.ndim == 3
            b = self.momentum
            eps = self.eps
            raw_ = (raw + eps).pow_(self.order)
            self.running_prop.lerp_(raw_/raw_.sum(dim=(-1, -2), keepdim=True), 1-b)
        return torch.inner(raw.ravel(), self.running_prop.ravel()) / self.num_channels
