
import sys
import argparse

sys.path.append('./src')

import yaml
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from unet_100 import build_model
from dataset import NPZDataset
from common import loss_fn, add_gaussian_noise


### parse args

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='config file.')

args = parser.parse_args()

with open(args.config, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

SAVE = config['save']
GPU_ID = config['gpu_id']
device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

iter_head: int = config['iter_head']
n_epoch: int = config['epochs']

lr = config['lr']
momentum = config['momentum']
weight_decay = config['weight_decay']

hc = config['high_cut']
noise = config.get('noise', 0.0)
s_trainable = config.get('s_trainable', True)

### build model & data set

model, MODEL_NAME = build_model(device, tag=config['tag'], hc=hc)
model.df_solver._frac.s.requires_grad_(s_trainable)

data_conf = config['data']
train_dataset = NPZDataset(data_conf['train_set_location'], data_conf['train_set_volume'])
validate_dataset = NPZDataset(data_conf['validate_set_location'], data_conf['validate_set_volume'])
loader = DataLoader(train_dataset, batch_size=data_conf['train_batch_size'], shuffle=True, num_workers=4, pin_memory=True)
loader_2 = DataLoader(validate_dataset, batch_size=data_conf['validate_batch_size'], shuffle=True, num_workers=1, pin_memory=True)

iter_per_epoch, remander = divmod(len(train_dataset), data_conf['train_batch_size'])
assert remander == 0

### confirm

print(f'\nStart training {MODEL_NAME} on {device}...')
print(f"  - s trainable: {s_trainable}")

print(f'Total {n_epoch} epochs(iter from {iter_head}), {iter_per_epoch} iterations per epoch.')
print(f'Training set size: {len(train_dataset)}, noise: {noise}.')
print(f'Validation set size: {len(validate_dataset)}.', end='\n\n')
print("Train(SGD) setup:")
print(f"  - learning rate: {lr}")
print(f"  - momentum: {momentum}")
print(f"  - weight decay: {weight_decay}", end='\n\n')

log_dir = config['log_dir']
if log_dir[-1] != '/':
    log_dir += '/'

print(f"Logs will be saved in {log_dir}")

checkpoint_path = ''

if SAVE:
    checkpoint_dir = config['checkpoint_dir']

    if checkpoint_dir[-1] != '/':
        checkpoint_dir += '/'

    checkpoint_path = checkpoint_dir + MODEL_NAME + '.pth'
    print(f"Checkpoints will be saved as {checkpoint_path}", end='\n\n')

else:
    print("Checkpoints saving disabled.", end='\n\n')

signal_ = input("Continue? (y/n)")

if signal_ not in {'y', 'Y'}:
    print("Aborted.")
    exit(0)

### train

optim = SGD(model.parameters(), lr=lr,
            momentum=momentum, weight_decay=weight_decay)

writer_1 = SummaryWriter(log_dir + MODEL_NAME, flush_secs=30)

if SAVE:
    import os
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

def train(epoch: int):
    step = 0

    for gdgn, label in tqdm(loader, desc=f'Epoch {epoch + 1}/{n_epoch}', unit='batch', leave=False):
        optim.zero_grad()

        add_gaussian_noise(gdgn[:, :, 0, :], noise)

        y_out = model(gdgn.to(device=device, non_blocking=True)) # (N, 1, Nx, Ny)
        loss = loss_fn(y_out, label.flatten().to(dtype=torch.float32, device=device))
        loss.backward()
        optim.step()
        step += 1

        writer_1.add_scalar('loss(train)', loss.item(),
                            iter_head + epoch*iter_per_epoch + step)
        writer_1.add_scalar('s', model.df_solver._frac.s.item(),
                            iter_head + epoch*iter_per_epoch + step)

    if SAVE:
        torch.save(model.state_dict(), checkpoint_path)


def validate(epoch):
    gdgn, label = next(iter(loader_2))
    with torch.no_grad():
        y_out = model(gdgn.to(device=device))
        loss = loss_fn(y_out, label.flatten().to(dtype=torch.float32, device=device))

    writer_1.add_scalar('loss(validate)', loss.item(), iter_head + (epoch + 1)*iter_per_epoch)


for epoch in trange(0, n_epoch, desc='Training', unit='epoch'):
    train(epoch)
    validate(epoch)

writer_1.close()
print("Done.")
