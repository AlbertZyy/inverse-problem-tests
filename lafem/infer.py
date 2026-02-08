
import os
import sys
import argparse

sys.path.append('./src')

import yaml
import numpy as np
import torch
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import RandomSampler, BatchSampler
from torch.nn.functional import binary_cross_entropy
# from kokomi import Arrange
from tqdm import tqdm, trange

from lafemeit.model import build_eit_model, Fractional
from lafemeit.utils import NPYDataset, NPZDataset, MemoryDataset
from bce_loss import DropedBCELoss



parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='config file.')

args = parser.parse_args()

with open(args.config, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


def main(noise: float, use_noise_filter: bool, tag: str, type_: str, gpu_id: int, **kwargs):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    data_conf = config['data']

    train_data_dataset = NPYDataset(
        os.path.join(data_conf['train_set_location'], 'gd'),
        names=[str(i) for i in range(data_conf['train_set_start'], data_conf['train_set_end'])],
    )
    train_data_dataset = MemoryDataset(
        train_data_dataset.names,
        train_data_dataset.read_data,
        num_workers=0,
        device=device,
        tqdm=True
    )

    train_label_dataset = NPZDataset(
        os.path.join(data_conf['train_set_location'], 'inclusion'),
        names = [str(i) for i in range(data_conf['train_set_start'], data_conf['train_set_end'])],
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
        os.path.join(data_conf['validate_set_location'], 'gd'),
        names=[str(i) for i in range(data_conf['validate_set_start'], data_conf['validate_set_end'])],
    )
    validate_data_dataset = MemoryDataset(
        validate_data_dataset.names,
        validate_data_dataset.read_data,
        num_workers=0,
        device=device,
        tqdm=True
    )

    validate_label_dataset = NPZDataset(
        os.path.join(data_conf['validate_set_location'], 'inclusion'),
        names = [str(i) for i in range(data_conf['validate_set_start'], data_conf['validate_set_end'])],
        channel_keys = []
    )
    validate_label_dataset = MemoryDataset(
        validate_label_dataset.names_seq,
        validate_label_dataset._read_data,
        num_workers=0,
        device=device,
        tqdm=True
    )

    gn_origin = torch.from_numpy(
        np.load(os.path.join(data_conf['train_set_location'], 'gn.npy'))
    ).to(device)


    model, MODEL_NAME = build_eit_model(
        name = 'unet100',
        ext = 63,
        n_channel = 8,
        tag = tag,
        fractype = type_,
        eigen_file = "lafem/data/laplace_beltrami_63_63.npz",
        ckpts_path = "lafem/ckpts",
        device = device
    )
    DATA_ID = 10001

    model.eval()
    sampler = RandomSampler(validate_data_dataset)
    batch_sampler = BatchSampler(sampler, batch_size=data_conf['validate_batch_size'], drop_last=False)
    losses = []

    with torch.no_grad():
        gd = validate_data_dataset[DATA_ID]
        gn = gn_origin[None, ...].broadcast_to(gd.shape)
        gdgn = torch.stack([gd, gn], dim=-2) # (N, CH, 2, bddof)

        noise_tensor = torch.randn_like(gdgn[:, :, 0, :]) * noise
        noise_tensor = gdgn[:, :, 0, :] * noise_tensor
        gdgn[:, :, 0, :] += noise_tensor

        y_out = model(gdgn).squeeze(1) # (N, 1, Nx, Ny)
        label = validate_label_dataset[DATA_ID].reshape(y_out.shape)
        loss = binary_cross_entropy(y_out, label.to(dtype=torch.float32))
