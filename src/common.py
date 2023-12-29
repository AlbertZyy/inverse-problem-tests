
import torch
from torch import Tensor


def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
    val = - torch.log(y_pred) * y_true - torch.log(1 - y_pred) * (1 - y_true)
    return torch.mean(val)
