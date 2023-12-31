
import torch
from torch import Tensor


def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
    val = - torch.log(y_pred) * y_true - torch.log(1 - y_pred) * (1 - y_true)
    return torch.mean(val)


def add_gaussian_noise(tensor: Tensor, mean=0., std=1.):
    """
    Adds multiplicative Gaussian noise to the given PyTorch tensor. This is an
    inplace operation.

    Parameters:
        tensor (Tensor): Input tensor.
        mean (float): Mean of the Gaussian distribution (default is 0).
        std (float): Standard deviation of the Gaussian distribution (default is 1).

    Returns:
        Tensor: Tensor with added Gaussian noise.
    """
    noise = torch.randn_like(tensor) * std + mean
    tensor += tensor * noise
    return tensor


if __name__ == '__main__':
    # Example usage
    input_tensor = torch.tensor([1.0, 2.0, 100.0])
    output_tensor = add_gaussian_noise(input_tensor, mean=0, std=0.1)

    print("Input Tensor:", input_tensor)
    print("Output Tensor with Gaussian Noise:", output_tensor)
