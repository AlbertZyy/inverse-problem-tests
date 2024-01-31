
import sys
import argparse

sys.path.append('./src')

import yaml
import torch
from torch.optim import SGD
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from dataset import TPZDataset
from cnn import build_model
from common import add_multi_std_gaussian_noise, total_variation


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
tv_scale: int   = config['tv_scale']


device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

train_dataset = TPZDataset(
    data_conf['train_set_location'],
    data_conf['train_set_volume'],
    channel_keys=data_conf['train_set_channels'],
    num_workers=0,
    tqdm=True
)
validate_dataset = TPZDataset(
    data_conf['validate_set_location'],
    data_conf['validate_set_volume'],
    channel_keys=data_conf['validate_set_channels'],
    num_workers=0,
    tqdm=True
)

iter_per_epoch, remander = divmod(len(train_dataset), data_conf['train_batch_size'])
assert remander == 0

model, MODEL_NAME = build_model(device, tag)


### confirm

print(f'\nStart training {MODEL_NAME} on {device}...')
print(f'Total {n_epoch} epochs(iter from {iter_head}), {iter_per_epoch} iterations per epoch.')
print(f'Training set size: {len(train_dataset)}.')
print(f'Validation set size: {len(validate_dataset)}.', end='\n\n')
print(f"SGD: {config['optim'].items()}\n")
print(f"TV scale: {tv_scale}")

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

EXT = 63
bd_index = torch.zeros((4*EXT, ), dtype=torch.int, device=device)
bd_index[0     : EXT]   = torch.arange(0, EXT)
bd_index[EXT   : 2*EXT] = torch.arange(EXT, 3*EXT, 2)
bd_index[2*EXT : 3*EXT] = torch.arange(4*EXT-1, 3*EXT-1, -1)
bd_index[3*EXT : ]      = torch.arange(3*EXT-1, EXT, -2)


optim = SGD(model.parameters(), **config['optim'])
loss_fn = torch.nn.MSELoss()
writer_1 = SummaryWriter(log_dir + MODEL_NAME, max_queue=1000, flush_secs=30)


if SAVE:
    import os
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

def train(epoch: int):
    step = 0

    for gdgn, _ in tqdm(train_dataset.loader(batch_size=data_conf['train_batch_size']),
                        desc=f'Epoch {epoch + 1}/{n_epoch}', unit='batch', leave=False):
        optim.zero_grad()
        BATCH, CHANNEL, _, BDDOF = gdgn.shape
        gdgn_ori = gdgn.clone().to(device=device, non_blocking=True)
        gdgn = gdgn.to(device=device, non_blocking=True)

        g = torch.randn((BATCH, ), device=device)
        noise = torch.exp(g) * 1e-2
        add_multi_std_gaussian_noise(gdgn[:, :, 0, :], noise)

        y_out = model(gdgn.reshape(BATCH, CHANNEL*2, BDDOF))
        mse = loss_fn(y_out, gdgn_ori.reshape(BATCH, CHANNEL*2, BDDOF))
        tv = total_variation(y_out[..., bd_index], boundary='circular')
        loss = mse + tv * tv_scale
        loss.backward()
        optim.step()
        step += 1

        writer_1.add_scalar('mse(train)', mse.item(),
                            iter_head + epoch*iter_per_epoch + step)
        writer_1.add_scalar('tv(train)', tv.item(),
                            iter_head + epoch*iter_per_epoch + step)
        writer_1.add_scalar('loss(train)', loss.item(),
                            iter_head + epoch*iter_per_epoch + step)

    if SAVE:
        torch.save(model.state_dict(), checkpoint_path)


def validate(epoch):
    gdgn, _ = next(validate_dataset.loader(batch_size=data_conf['validate_batch_size']))
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
