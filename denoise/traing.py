
import sys
import argparse

sys.path.append('./src')

import yaml
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from dataset import NPZDataset
from cnn import build_model
from fractional import Fractional
from common import add_multi_std_gaussian_noise


parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, default='config.yaml')
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


tag: str        = config.get('tag', '')
data_conf       = config['data']
SAVE            = config['save']
GPU_ID          = config['gpu_id']
iter_head: int  = config['iter_head']
n_epoch: int    = config['epochs']
lr              = config['lr']
momentum        = config.get('momentum', 0)
weight_decay    = config.get('weight_decay', 0.0)


device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

train_dataset = NPZDataset(data_conf['train_set_location'], data_conf['train_set_volume'])
validate_dataset = NPZDataset(data_conf['validate_set_location'], data_conf['validate_set_volume'])
loader_1 = DataLoader(train_dataset, batch_size=data_conf['train_batch_size'],
                      shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=4)
loader_2 = DataLoader(validate_dataset, batch_size=data_conf['validate_batch_size'],
                      shuffle=True, num_workers=2, pin_memory=True)
iter_per_epoch, remander = divmod(len(train_dataset), data_conf['train_batch_size'])
assert remander == 0

model, MODEL_NAME = build_model(device, tag)


### confirm

print(f'\nStart training {MODEL_NAME} on {device}...')
print(f'Total {n_epoch} epochs(iter from {iter_head}), {iter_per_epoch} iterations per epoch.')
print(f'Training set size: {len(train_dataset)}.')
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

### training

optim = SGD(model.parameters(), lr=lr,
            momentum=momentum, weight_decay=weight_decay)
loss_fn = torch.nn.MSELoss()
writer_1 = SummaryWriter(log_dir + MODEL_NAME, flush_secs=30)
filter_ = Fractional(252, device=device)
filter_.from_npz(f"./data/laplace_beltrami_{63}_{63}.npz")
filter_.initialize(0.75)
filter_.s.requires_grad_(False)

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

    for gdgn, _ in tqdm(loader_1, desc=f'Epoch {epoch + 1}/{n_epoch}', unit='batch', leave=False):
        optim.zero_grad()
        BATCH, CHANNEL, _, BDDOF = gdgn.shape
        gdgn_ori = gdgn.clone().to(device=device, non_blocking=True)
        gdgn = gdgn.to(device=device, non_blocking=True)

        g = torch.randn((BATCH, ), device=device)
        noise = torch.exp(g) * 1e-2
        add_multi_std_gaussian_noise(gdgn[:, :, 0, :], noise)

        y_out = model(filter_(gdgn.reshape(BATCH, CHANNEL*2, BDDOF)))
        loss = loss_fn(y_out,
                       filter_(gdgn_ori.reshape(BATCH, CHANNEL*2, BDDOF)))
        loss.backward()
        optim.step()
        step += 1

        writer_1.add_scalar('loss(train)', loss.item(),
                            iter_head + epoch*iter_per_epoch + step)

    if SAVE:
        torch.save(model.state_dict(), checkpoint_path)


def validate(epoch):
    gdgn, _ = next(iter(loader_2))
    BATCH, CHANNEL, _, BDDOF = gdgn.shape
    with torch.no_grad():
        gdgn = gdgn.to(device=device, non_blocking=True)
        gdgn_ori = gdgn.clone().to(device=device, non_blocking=True)
        g = torch.randn((BATCH, ), device=device)
        noise = torch.exp(g) * 1e-2
        add_multi_std_gaussian_noise(gdgn[:, :, 0, :], noise)

        y_out = model(gdgn.reshape(BATCH, CHANNEL*2, BDDOF))
        loss = loss_fn(y_out, gdgn_ori.reshape(BATCH, CHANNEL*2, BDDOF))

    writer_1.add_scalar('loss(validate)', loss.item(), iter_head + (epoch + 1)*iter_per_epoch)


for epoch in trange(0, n_epoch, desc='Training', unit='epoch'):
    train(epoch)
    validate(epoch)

writer_1.close()
print("Done.")
