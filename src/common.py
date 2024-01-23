
from typing import Dict, Any, Iterable

import torch
from torch import Tensor
from torch.nn import Module


OPTIM_MAP = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam,
}

def load_optimizer(config_dict: Dict[str, Any], params: Iterable[Tensor]) -> torch.optim.Optimizer:
    options = config_dict.copy()
    opt_name: str = config_dict['optimizer']
    del options['optimizer']

    optim_class = OPTIM_MAP.get(opt_name, None)
    if optim_class is None:
        raise ValueError(f'Unsupported optimizer: {opt_name}')

    return optim_class(params=params, **options)


def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
    val = - torch.log(y_pred) * y_true - torch.log(1 - y_pred) * (1 - y_true)
    return torch.mean(val)


def add_gaussian_noise(tensor: Tensor, std=1.):
    """
    Adds multiplicative Gaussian noise to the given PyTorch tensor. In-place operation.

    Parameters:
        tensor (Tensor): Input tensor.
        std (float): Standard deviation of the Gaussian distribution (default is 1).

    Returns:
        Tensor: Tensor with added Gaussian noise.
    """
    noise = torch.randn_like(tensor) * std + 1.
    tensor.copy_(tensor * noise)
    return tensor


def add_filtered_gaussian_noise(filter: Module, tensor: Tensor, std=1.) -> Tensor:
    """
    Adds multiplicative Gaussian noise to the given PyTorch tensor. In-place operation.

    Parameters:
        filter (Module): Filter module.
        tensor (Tensor): Input tensor.

    Returns:
        Tensor: Tensor with added Gaussian noise.
    """
    noise = filter(torch.randn_like(tensor)) * std + 1.
    tensor.copy_(tensor * noise)
    return tensor


def add_multi_std_gaussian_noise(tensor: Tensor, std: Tensor) -> Tensor:
    """
    Adds multiplicative Gaussian noise to the given PyTorch tensor. In-place operation.

    Parameters:
        tensor (Tensor): Input tensor with shape [N, ...].
        std (Tensor): 1-d Standard deviation Tensor of the Gaussian distribution\
            for each sample, with shape [N,].

    Returns:
        Tensor: Tensor with added Gaussian noise.
    """
    raw = torch.randn_like(tensor[0, ...])
    noise = torch.einsum('..., c -> c...', raw, std) + 1.
    tensor.copy_(tensor * noise)
    return tensor


if __name__ == '__main__':
    # Example usage
    input_tensor = torch.tensor([1.0, 2.0, 100.0])
    output_tensor = add_gaussian_noise(input_tensor, std=0.1, noise_only=True)

    print("Input Tensor:", input_tensor)
    print("Output Tensor with Gaussian Noise:", output_tensor)
