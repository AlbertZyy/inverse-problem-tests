from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def randomized_measurement_subbatch(
    data: Tensor,
    min_L: int = 2,
    max_L: Optional[int] = None,
) -> Tensor:
    """
    Apply randomized batching over the measurement axis.

    Parameters
    ----------
    data:
        Input measurements of shape [B, L, F].
    min_L:
        Minimum number of measurements to keep.
    max_L:
        Maximum number of measurements to keep. If None, use the full L.

    Returns
    -------
    sub_data:
        Tensor of shape [B, L_hat, F], where L_hat is randomly sampled
        once per training step, and each sample in the batch gets its own
        random subset of measurements.
    """
    if data.ndim != 3:
        raise ValueError(f"`data` must have shape [B, L, F], got {tuple(data.shape)}.")

    batch_size, total_L, feature_dim = data.shape
    del feature_dim

    if total_L < 1:
        raise ValueError("Measurement dimension L must be at least 1.")

    if max_L is None:
        max_L = total_L

    max_L = min(max_L, total_L)
    min_L = min(min_L, max_L)

    if min_L < 1:
        raise ValueError(f"`min_L` must be >= 1, got {min_L}.")

    # Sample one L_hat for the whole mini-batch, which keeps tensor shapes uniform.
    L_hat = torch.randint(
        low=min_L,
        high=max_L + 1,
        size=(1,),
        device=data.device,
    ).item()

    # Independent random subset per sample.
    # scores: [B, L], take the smallest/top-k indices after sorting.
    scores = torch.rand(batch_size, total_L, device=data.device)
    selected_idx = scores.argsort(dim=1)[:, :L_hat]  # [B, L_hat]

    gather_idx = selected_idx.unsqueeze(-1).expand(-1, -1, data.shape[-1])
    sub_data = torch.gather(data, dim=1, index=gather_idx)
    return sub_data


def add_multiplicative_gaussian_noise(
    data: Tensor,
    noise_std: float,
) -> Tensor:
    """
    Add multiplicative Gaussian noise:
        x_noisy = x * (1 + noise_std * eps), eps ~ N(0, 1)

    Parameters
    ----------
    data:
        Input tensor.
    noise_std:
        Standard deviation of the multiplicative Gaussian noise.

    Returns
    -------
    noisy_data:
        Tensor with multiplicative noise added.
    """
    if noise_std < 0.0:
        raise ValueError(f"`noise_std` must be non-negative, got {noise_std}.")

    if noise_std == 0.0:
        return data

    noise = torch.randn_like(data)
    return data * (1.0 + noise_std * noise)


def forward_flattened_prediction(
    model: torch.nn.Module,
    data: Tensor,
    coords: Tensor,
    grid_size: Optional[Tuple[int, int]] = None,
) -> Tensor:
    """
    Forward NIO model and flatten the output to [B, num_points].

    Returns
    -------
    pred_flat:
        Tensor of shape [B, num_points]
    """
    pred = model(data, coords, grid_size=grid_size)  # [B, 1, H, W]
    if pred.ndim != 4 or pred.shape[1] != 1:
        raise ValueError(
            "Expected model output shape [B, 1, H, W], "
            f"got {tuple(pred.shape)}."
        )
    pred_flat = pred.flatten(start_dim=1)  # [B, H*W]
    return pred_flat