import os
import sys
sys.path.append('nio')

import torch
from torch import Tensor
from typing import Optional, Tuple
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, StackDataset
from torch.utils.tensorboard.writer import SummaryWriter
from lafemeit.utils import NPYDataset, NPZDataset, MemoryDataset


from ops import (
    randomized_measurement_subbatch,
    add_multiplicative_gaussian_noise,
    forward_flattened_prediction
)


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    coords: Tensor,
    device: torch.device,
    noise_std: float,
    min_randomized_L: int,
    max_randomized_L: Optional[int] = None,
    grid_size: Optional[Tuple[int, int]] = None,
    max_grad_norm: Optional[float] = None,
) -> float:
    """
    Train the model for one epoch.

    Parameters
    ----------
    model:
        NIO model.
    train_loader:
        Dataloader yielding (data, label), with:
            data:  [B, L, F]
            label: [B, N]
    optimizer:
        Optimizer.
    coords:
        Coordinate grid used by trunk / NIO forward.
    device:
        Training device.
    noise_std:
        Multiplicative Gaussian noise level.
    min_randomized_L:
        Minimum number of measurements kept in randomized batching.
    max_randomized_L:
        Maximum number of measurements kept in randomized batching.
    grid_size:
        Required if coords is flattened as [N, 2].
    max_grad_norm:
        If not None, apply gradient clipping.

    Returns
    -------
    mean_loss:
        Mean training loss over the epoch.
    """
    model.train()

    coords = coords.to(device, non_blocking=True)

    total_loss = 0.0
    total_samples = 0

    for data, label in train_loader:
        data = data.to(dtype=torch.float32, device=device, non_blocking=True)    # [B, L, F]
        label = label.to(dtype=torch.float32, device=device, non_blocking=True)  # [B, N]

        # Randomized batching on the measurement axis.
        data = randomized_measurement_subbatch(
            data=data,
            min_L=min_randomized_L,
            max_L=max_randomized_L,
        )

        # Multiplicative Gaussian noise on training inputs only.
        data = add_multiplicative_gaussian_noise(
            data=data,
            noise_std=noise_std,
        )

        optimizer.zero_grad(set_to_none=True)

        pred_flat = forward_flattened_prediction(
            model=model,
            data=data,
            coords=coords,
            grid_size=grid_size,
        )  # [B, N]

        if pred_flat.shape != label.shape:
            raise ValueError(
                f"Prediction/label shape mismatch: "
                f"pred={tuple(pred_flat.shape)}, label={tuple(label.shape)}."
            )

        loss = F.binary_cross_entropy_with_logits(pred_flat, label)

        loss.backward()

        if max_grad_norm is not None:
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        batch_size = data.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    mean_loss = total_loss / max(total_samples, 1)
    return mean_loss


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    coords: Tensor,
    device: torch.device,
    noise_std: float = 0.0,
    grid_size: Optional[Tuple[int, int]] = None,
) -> float:
    """
    Evaluate the model for one epoch.

    Parameters
    ----------
    model:
        NIO model.
    val_loader:
        Dataloader yielding (data, label), with:
            data:  [B, L, F]
            label: [B, N]
    coords:
        Coordinate grid used by trunk / NIO forward.
    device:
        Validation device.
    grid_size:
        Required if coords is flattened as [N, 2].

    Returns
    -------
    mean_loss:
        Mean validation loss over the epoch.
    """
    model.eval()

    coords = coords.to(device, non_blocking=True)

    total_loss = 0.0
    total_samples = 0

    for data, label in val_loader:
        data = data.to(dtype=torch.float32, device=device, non_blocking=True)    # [B, L, F]
        label = label.to(dtype=torch.float32, device=device, non_blocking=True)  # [B, N]

        data = add_multiplicative_gaussian_noise(
            data=data,
            noise_std=noise_std,
        )

        pred_flat = forward_flattened_prediction(
            model=model,
            data=data,
            coords=coords,
            grid_size=grid_size,
        )

        if pred_flat.shape != label.shape:
            raise ValueError(
                f"Prediction/label shape mismatch: "
                f"pred={tuple(pred_flat.shape)}, label={tuple(label.shape)}."
            )

        loss = F.binary_cross_entropy_with_logits(pred_flat, label)

        batch_size = data.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    mean_loss = total_loss / max(total_samples, 1)
    return mean_loss


import sucrose

from model import NIO2d
from fno import FNO2d
from deeponet import DeepONet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ssc = sucrose.scenario("nio", "base")
    deeponet_model = ssc.partial(DeepONet, "deeponet")()
    fno_model = ssc.partial(FNO2d, "fno2d")()
    model = NIO2d(
        deeponet_model, fno_model,
        d_v=ssc.get_config("nio.d_v"),
        geometry_dim=ssc.get_config("nio.geometry_dim")
    )

    # count number of parameters in these three models:
    num_params_deeponet = sum(p.numel() for p in deeponet_model.parameters())
    num_params_fno = sum(p.numel() for p in fno_model.parameters())
    num_params_nio = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: DeepONet={num_params_deeponet}, "
          f"FNO={num_params_fno}, NIO={num_params_nio}")

    model = model.to(device)

    WIDTH = ssc.get_config("data.width")
    HEIGHT = ssc.get_config("data.height")
    # 生成坐标网格，形状为 [H, W, 2]
    x = torch.linspace(-1, 1, WIDTH)
    y = torch.linspace(-1, 1, HEIGHT)
    xv, yv = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack((xv, yv), dim=-1)  # [H, W, 2]
    coords = coords.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ssc.get_config("optim.lr"),
        weight_decay=ssc.get_config("optim.weight_decay"),
    )
    ssc.load_state_dict(
        model=model,
        optimizer=optimizer,
    )

    writer = ssc.start_pytorch_tensorboard()
    num_epochs = ssc.get_config("training.num_epochs")


    train_data_dataset = NPYDataset(
        os.path.join(ssc.get_config('data.train_set_location'), 'gd'),
        names=[str(i) for i in range(ssc.get_config('data.train_set_start'), ssc.get_config('data.train_set_end'))],
    )
    train_data_dataset = MemoryDataset(
        train_data_dataset.names,
        train_data_dataset.read_data,
        num_workers=0,
        device=device,
        tqdm=True
    )

    train_label_dataset = NPZDataset(
        os.path.join(ssc.get_config('data.train_set_location'), 'inclusion'),
        names = [str(i) for i in range(ssc.get_config('data.train_set_start'), ssc.get_config('data.train_set_end'))],
        channel_keys = []
    )
    train_label_dataset = MemoryDataset(
        train_label_dataset.names_seq,
        train_label_dataset._read_data,
        num_workers=0,
        device=device,
        tqdm=True
    )

    validate_data_dataset = NPYDataset(
        os.path.join(ssc.get_config('data.val_set_location'), 'gd'),
        names=[str(i) for i in range(ssc.get_config('data.val_set_start'), ssc.get_config('data.val_set_end'))],
    )
    validate_data_dataset = MemoryDataset(
        validate_data_dataset.names,
        validate_data_dataset.read_data,
        num_workers=0,
        device=device,
        tqdm=True
    )

    validate_label_dataset = NPZDataset(
        os.path.join(ssc.get_config('data.val_set_location'), 'inclusion'),
        names = [str(i) for i in range(ssc.get_config('data.val_set_start'), ssc.get_config('data.val_set_end'))],
        channel_keys = []
    )
    validate_label_dataset = MemoryDataset(
        validate_label_dataset.names_seq,
        validate_label_dataset._read_data,
        num_workers=0,
        device=device,
        tqdm=True
    )

    training_set = StackDataset(train_data_dataset, train_label_dataset)
    val_set = StackDataset(validate_data_dataset, validate_label_dataset)

    train_loader = DataLoader(training_set, batch_size=ssc.get_config("training.batch_size"), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=ssc.get_config("training.batch_size"), shuffle=False)

    for epoch in ssc.epoch_range(num_epochs):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            coords=coords,
            device=device,
            noise_std=ssc.get_config("data.noise_std"),
            min_randomized_L=2,
            max_randomized_L=8,
            grid_size=(64, 64),
            max_grad_norm=None,
        )

        val_loss = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            coords=coords,
            device=device,
            noise_std=ssc.get_config("data.noise_std"),
            grid_size=(64, 64),
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("lr", current_lr, epoch)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"lr={current_lr:.3e}"
        )

        ssc.save_state_dict(
            interval=100,
            model=model, # contain all sub-models' parameters
            optimizer=optimizer,
        )

    writer.close()


if __name__ != "__main__":
    exit(0)

main()
