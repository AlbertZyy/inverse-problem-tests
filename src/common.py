
import torch
from torch import Tensor


def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
    val = - torch.log(y_pred) * y_true - torch.log(1 - y_pred) * (1 - y_true)
    return torch.mean(val)


def add_gaussian_noise(tensor: Tensor, std=1., noise_only=False):
    """
    Adds multiplicative Gaussian noise to the given PyTorch tensor. In-place operation.

    Parameters:
        tensor (Tensor): Input tensor.
        std (float): Standard deviation of the Gaussian distribution (default is 1).

    Returns:
        Tensor: Tensor with added Gaussian noise.
    """
    noise = torch.randn_like(tensor)
    noise *= std

    if noise_only:
        return tensor * noise
    else:
        torch.mul(tensor, noise + 1., out=tensor)
        return tensor


if __name__ == '__main__':
    # Example usage
    input_tensor = torch.tensor([1.0, 2.0, 100.0])
    output_tensor = add_gaussian_noise(input_tensor, std=0.1, noise_only=True)

    print("Input Tensor:", input_tensor)
    print("Output Tensor with Gaussian Noise:", output_tensor)
