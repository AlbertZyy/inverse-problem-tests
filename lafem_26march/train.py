
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
from kokomi import Arrange
from tqdm import tqdm, trange

from lafemeit.model import build_eit_model, FractionalDoF
from lafemeit.utils import NPYDataset, NPZDataset, MemoryDataset

### parse args

COLOR = {
    0: '#ff0000',
    1: '#ff7f00',
    2: '#ffff00',
    3: '#00ff00',
    4: '#00ffff',
    5: '#0000ff',
    6: '#ff00ff'
}

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='config file.')

args = parser.parse_args()

with open(args.config, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


data_conf       = config['data']
SAVE            = config['save']
iter_head: int  = config['iter_head']
n_epoch: int    = config['epochs']
lr              = config['lr']
momentum        = config.get('momentum', 0)
weight_decay    = config.get('weight_decay', 0.0)


works = Arrange()
works.import_yaml(args.config)
print(works._pool_size)

print("Train(SGD) setup:")
print(f"  - learning rate: {lr}")
print(f"  - momentum: {momentum}")
print(f"  - weight decay: {weight_decay}", end='\n\n')


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
        eigen_file = "data/laplace_beltrami_64x64.npz",
        ckpts_path = "lafem_26march/ckpts",
        device = device
    )
    iter_per_epoch, remander = divmod(len(train_data_dataset), data_conf['train_batch_size'])
    assert remander == 0

    if use_noise_filter:
        noise_filter = FractionalDoF(252, device=device)
        noise_filter.from_npz("data/laplace_beltrami_64x64.npz")
        noise_filter.initialize(gamma=-0.75)
        noise_filter.gamma.requires_grad_(False)
    else:
        noise_filter = None

    ### confirm

    # print(f'\nStart training {MODEL_NAME} on {device}...')
    # print(f"  - type: {type_}")
    # print(f'Training set size: {len(train_data_dataset)}.')
    # print(f'Total {n_epoch} epochs(iter from {iter_head}), {iter_per_epoch} iterations per epoch.')
    # print(f"NOISE: {noise}, filter: {use_noise_filter}.")

    log_dir = config['log_dir']

    # print(f"Logs will be saved in {log_dir}")

    if SAVE:
        checkpoint_dir = config['checkpoint_dir']
        # print(f"Checkpoints will be saved as {checkpoint_path}", end='\n\n')
        checkpoint_path = os.path.join(checkpoint_dir, MODEL_NAME + '.pth')
    else:
        checkpoint_path = ''
        print("Checkpoints saving disabled.", end='\n\n')

    ### train

    optim = SGD(model.parameters(), lr=lr,
                momentum=momentum, weight_decay=weight_decay)
    writer_1 = SummaryWriter(os.path.join(log_dir, MODEL_NAME), flush_secs=30)

    if SAVE:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    for epoch in trange(0, n_epoch, desc=f'Device: {gpu_id}', ascii=True,
                        unit='epoch', leave=False, position=gpu_id*2,
                        colour=COLOR.get(gpu_id, 'white')):
        ### train
        model.train()
        step = 0
        sampler = RandomSampler(train_data_dataset)
        batch_sampler = BatchSampler(sampler, batch_size=data_conf['train_batch_size'], drop_last=False)

        for indices in tqdm(batch_sampler,
                            desc=f'  Epoch {epoch + 1}/{n_epoch}', ascii=True,
                            unit='batch', leave=False, position=gpu_id*2+1):
            gd = train_data_dataset[indices]
            gn = gn_origin[None, ...].broadcast_to(gd.shape)
            gdgn = torch.stack([gd, gn], dim=-2) # (N, CH, 2, bddof)
            optim.zero_grad()

            noise_tensor = torch.randn_like(gdgn[:, :, 0, :]) * noise
            if noise_filter:
                noise_tensor = noise_filter(noise_tensor)
            noise_tensor = gdgn[:, :, 0, :] * noise_tensor
            gdgn[:, :, 0, :] += noise_tensor

            y_out = model(gdgn).squeeze(1) # (N, 1, Nx, Ny)
            label = train_label_dataset[indices].reshape(y_out.shape)
            loss = binary_cross_entropy(y_out, label.to(dtype=torch.float32))
            loss.backward()
            optim.step()
            step += 1

        writer_1.add_scalar('loss(train)', loss.item(),
                            iter_head + (epoch+1)*iter_per_epoch)

        if type_ != 'nograd':
            if type_ != 'single':
                for i in range(0, 8):
                    writer_1.add_scalar(f'gamma{i}', model.df_solver.bc_filter.gamma[i].item(),
                                        iter_head + (epoch+1)*iter_per_epoch)
            else:
                writer_1.add_scalar('gamma', model.df_solver.bc_filter.gamma.item(),
                                    iter_head + (epoch+1)*iter_per_epoch)

        if SAVE:
            torch.save(model.state_dict(), checkpoint_path)

        ### validate
        model.eval()
        sampler = RandomSampler(validate_data_dataset)
        batch_sampler = BatchSampler(sampler, batch_size=data_conf['validate_batch_size'], drop_last=False)
        losses = []

        with torch.no_grad():
            for indices in tqdm(batch_sampler,
                            desc=f'Epoch {epoch + 1}/{n_epoch}', ascii=True,
                            unit='batch', leave=False, position=gpu_id*2+1):
                gd = validate_data_dataset[indices]
                gn = gn_origin[None, ...].broadcast_to(gd.shape)
                gdgn = torch.stack([gd, gn], dim=-2) # (N, CH, 2, bddof)

                noise_tensor = torch.randn_like(gdgn[:, :, 0, :]) * noise
                if noise_filter:
                    noise_tensor = noise_filter(noise_tensor)
                noise_tensor = gdgn[:, :, 0, :] * noise_tensor
                gdgn[:, :, 0, :] += noise_tensor

                y_out = model(gdgn).squeeze(1) # (N, 1, Nx, Ny)
                label = validate_label_dataset[indices].reshape(y_out.shape)
                loss = binary_cross_entropy(y_out, label.to(dtype=torch.float32))
                losses.append(loss.item())

            loss_mean = sum(losses) / len(losses)

        writer_1.add_scalar('loss(validate)', loss_mean, iter_head + (epoch + 1)*iter_per_epoch)

    writer_1.close()


if __name__ == '__main__':
    works.run(main)
    # main(noise=0.0, use_noise_filter=False, tag='nn_multi', type_='multi', gpu_id=0)
