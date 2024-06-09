
import sys
import argparse

sys.path.append('./src')

import yaml
import torch
from torch.optim import SGD
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import RandomSampler, BatchSampler
from tqdm import tqdm, trange

from fractional import Fractional
from unet_100 import build_model
from dataset import TPYDataset, TPZDataset
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
momentum               = config.get('momentum', 0)
weight_decay           = config.get('weight_decay', 0.0)
NOISE: float           = config.get('noise', 0.0)
use_noise_filter: bool = config.get('use_noise_filter', False)
type_: bool            = config.get('type_', '')

device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')


### build model & data set

model, MODEL_NAME = build_model(device, tag=config['tag'], type_=type_)

data_conf = config['data']
# train_dataset = NPZDataset(data_conf['train_set_location'], data_conf['train_set_volume'], use_cache=True)
# validate_dataset = NPZDataset(data_conf['validate_set_location'], data_conf['validate_set_volume'], use_cache=True)
# loader = DataLoader(train_dataset, batch_size=data_conf['train_batch_size'], shuffle=True, num_workers=4, pin_memory=True)
# loader_2 = DataLoader(validate_dataset, batch_size=data_conf['validate_batch_size'], shuffle=True, num_workers=1, pin_memory=True)

train_data_dataset = TPYDataset(
    data_conf['train_set_location']['data'],
    names=[f"gnvn_{i}" for i in range(data_conf['train_set_volume'])],
    num_workers=0,
    device=device,
    tqdm=True
)

train_label_dataset = TPZDataset(
    data_conf['train_set_location']['label'],
    names=[f"gd_{i}" for i in range(data_conf['train_set_volume'])],
    num_workers=0,
    device=device,
    tqdm=True
)

# validate_dataset = TPZDataset(
#     data_conf['validate_set_location'],
#     names=data_conf['validate_set_volume'],
#     channel_keys=['1', '2', '3', '4', '5', '6', '8', '16'],
#     num_workers=4,
#     device=device,
#     tqdm=True
# )

iter_per_epoch, remander = divmod(len(train_data_dataset), data_conf['train_batch_size'])
assert remander == 0

if use_noise_filter:
    noise_filter = Fractional(252, device=device)
    noise_filter.from_npz(r"./data/laplace_beltrami_torch_63_63.npz")
    noise_filter.initialize(s=-0.75)
    noise_filter.s.requires_grad_(False)
else:
    noise_filter = None

### confirm

print(f'\nStart training {MODEL_NAME} on {device}...')
print(f"  - type: {type_}")

print(f'Total {n_epoch} epochs(iter from {iter_head}), {iter_per_epoch} iterations per epoch.')
print(f'Training set size: {len(train_data_dataset)}.')
# print(f'Validation set size: {len(validate_dataset)}.', end='\n\n')
print(f"NOISE: {NOISE}, filter: {use_noise_filter}.")
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

# writer_1 = SummaryWriter(log_dir + MODEL_NAME, flush_secs=30)

if SAVE:
    import os
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

def train(epoch: int):
    step = 0
    sampler = RandomSampler(train_data_dataset)
    batch_sampler = BatchSampler(sampler, batch_size=data_conf['train_batch_size'], drop_last=False)

    for indices in tqdm(batch_sampler,
                        desc=f'Epoch {epoch + 1}/{n_epoch}', unit='batch', leave=False):
        gnvn = train_data_dataset[indices]
        optim.zero_grad()

        # noise = torch.randn_like(gdgn[:, :, 0, :]) * NOISE
        # if noise_filter:
        #     noise = noise_filter(noise)
        # noise = gdgn[:, :, 0, :] * noise
        # gdgn[:, :, 0, :] += noise

        y_out = model(gnvn).squeeze(1) # (N, 1, Nx, Ny)
        label = train_label_dataset[indices].reshape(y_out.shape)
        loss = loss_fn(y_out, label.to(dtype=torch.float32))
        loss.backward()
        optim.step()
        step += 1

    # writer_1.add_scalar('loss(train)', loss.item(),
    #                     iter_head + (epoch+1)*iter_per_epoch)

    # if type_ != 'sng':
    #     if type_ != 'single':
    #         for i in range(0, 8):
    #             writer_1.add_scalar(f's{i}', model.df_solver._frac.s[i].item(),
    #                                 iter_head + (epoch+1)*iter_per_epoch)
    #     else:
    #         writer_1.add_scalar('s', model.df_solver._frac.s.item(),
    #                             iter_head + (epoch+1)*iter_per_epoch)

    if SAVE:
        torch.save(model.state_dict(), checkpoint_path)


# def validate(epoch):
#     gdgn, label = next(validate_dataset.loader(data_conf['validate_batch_size']))
#     with torch.no_grad():

#         noise = torch.randn_like(gdgn[:, :, 0, :]) * NOISE
#         if noise_filter:
#             noise = noise_filter(noise)
#         noise = gdgn[:, :, 0, :] * noise
#         gdgn[:, :, 0, :] += noise

#         y_out = model(gdgn.to(device=device))
#         loss = loss_fn(y_out, label.flatten().to(dtype=torch.float32, device=device))

#     writer_1.add_scalar('loss(validate)', loss.item(), iter_head + (epoch + 1)*iter_per_epoch)


for epoch in trange(0, n_epoch, desc='Training', unit='epoch'):
    train(epoch)
    # validate(epoch)

# writer_1.close()
print("Done.")
