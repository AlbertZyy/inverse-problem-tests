
import sys
import argparse

sys.path.append('./src')

import yaml
import torch
from torch.optim import SGD
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from fractional import Fractional
from unet_100 import build_model
from dataset import TPZDataset
from common import loss_fn


### parse args

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='config file.')

args = parser.parse_args()

with open(args.config, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

tag: str        = config.get('tag', '')
data_conf       = config['data']
SAVE            = config['save']
GPU_ID          = config['gpu_id']
iter_head: int  = config['iter_head']
n_epoch: int    = config['epochs']
lr              = config['lr']
momentum        = config.get('momentum', 0)
weight_decay    = config.get('weight_decay', 0.0)
s_trainable     = config['s_trainable']
NOISE: float    = config.get('noise', 0.0)
multi_s: bool   = config.get('multi_s', False)

device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')


### build model & data set

model, MODEL_NAME = build_model(device, tag=config['tag'], multi_s=multi_s)
model.df_solver._frac.s.requires_grad_(s_trainable)

data_conf = config['data']
# train_dataset = NPZDataset(data_conf['train_set_location'], data_conf['train_set_volume'], use_cache=True)
# validate_dataset = NPZDataset(data_conf['validate_set_location'], data_conf['validate_set_volume'], use_cache=True)
# loader = DataLoader(train_dataset, batch_size=data_conf['train_batch_size'], shuffle=True, num_workers=4, pin_memory=True)
# loader_2 = DataLoader(validate_dataset, batch_size=data_conf['validate_batch_size'], shuffle=True, num_workers=1, pin_memory=True)

train_dataset = TPZDataset(
    data_conf['train_set_location'],
    names=data_conf['train_set_volume'],
    channel_keys=['1', '2', '3', '4', '5', '6', '8', '16'],
    num_workers=4,
    device=device,
    tqdm=True
)

validate_dataset = TPZDataset(
    data_conf['validate_set_location'],
    names=data_conf['validate_set_volume'],
    channel_keys=['1', '2', '3', '4', '5', '6', '8', '16'],
    num_workers=4,
    device=device,
    tqdm=True
)

iter_per_epoch, remander = divmod(len(train_dataset), data_conf['train_batch_size'])
assert remander == 0

noise_filter = Fractional(252, device=device)
noise_filter.from_npz(r"./data/laplace_beltrami_63_63.npz")
noise_filter.initialize(s=-0.75)
noise_filter.s.requires_grad_(False)

### confirm

print(f'\nStart training {MODEL_NAME} on {device}...')
print(f"  - s trainable: {s_trainable}")
print(f"  - multiple s: {multi_s}")

print(f'Total {n_epoch} epochs(iter from {iter_head}), {iter_per_epoch} iterations per epoch.')
print(f'Training set size: {len(train_dataset)}, noise: {NOISE}.')
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

    for gdgn, label in tqdm(train_dataset.loader(data_conf['train_batch_size']),
                            desc=f'Epoch {epoch + 1}/{n_epoch}', unit='batch', leave=False):
        optim.zero_grad()

        noise = torch.randn_like(gdgn[:, :, 0, :]) * NOISE
        noise = noise_filter(noise)
        noise = gdgn[:, :, 0, :] * noise
        gdgn[:, :, 0, :] += noise

        y_out = model(gdgn.to(device=device, non_blocking=True)) # (N, 1, Nx, Ny)
        loss = loss_fn(y_out, label.flatten().to(dtype=torch.float32, device=device))
        loss.backward()
        optim.step()
        step += 1

        writer_1.add_scalar('loss(train)', loss.item(),
                            iter_head + epoch*iter_per_epoch + step)
        if multi_s:
            for i in range(0, 8):
                writer_1.add_scalar(f's{i}', model.df_solver._frac.s[i].item(),
                                    iter_head + epoch*iter_per_epoch + step)
        else:
            writer_1.add_scalar('s', model.df_solver._frac.s.item(),
                                iter_head + epoch*iter_per_epoch + step)

    if SAVE:
        torch.save(model.state_dict(), checkpoint_path)


def validate(epoch):
    gdgn, label = next(validate_dataset.loader(data_conf['validate_batch_size']))
    with torch.no_grad():
        y_out = model(gdgn.to(device=device))
        loss = loss_fn(y_out, label.flatten().to(dtype=torch.float32, device=device))

    writer_1.add_scalar('loss(validate)', loss.item(), iter_head + (epoch + 1)*iter_per_epoch)


for epoch in trange(0, n_epoch, desc='Training', unit='epoch'):
    train(epoch)
    validate(epoch)

writer_1.close()
print("Done.")
