
import os
import sys
from typing import Callable, Optional
from functools import reduce

sys.path.append('./src')

import torch
from torch import Tensor
from torch.nn import Module, BCELoss
from tqdm import tqdm

# from common import loss_fn as cross_entropy
from fractional import Fractional
from unet_100 import build_model
from dataset import TPZDataset


low_pass = Fractional(252, device='cpu')
low_pass.from_npz(r"./data/laplace_beltrami_63_63.npz")
low_pass.initialize(s=-0.75)
low_pass.s.requires_grad_(False)


settings = [
#   ('tag',       'type', 'noise', 'filter', 'ckpts_path')
    ('nn_sng',      'sng',    0.0,  None, 'test_no_noise/ckpts'),
    ('nn_single',   'single', 0.0,  None, 'test_no_noise/ckpts'),
    ('nn_multi',    '',       0.0,  None, 'test_no_noise/ckpts'),
    ('nn_ad',       'ad',     0.0,  None, 'test_no_noise/ckpts'),

    ('gn01_sng',    'sng',    0.01, None, 'test_sp_ad/ckpts'),
    ('gn01_single', 'single', 0.01, None, 'test_sp_ad/ckpts'),
    ('gn01_multi',  '',       0.01, None, 'test_sp_ad/ckpts'),
    ('gn01_ad',     'ad',     0.01, None, 'test_sp_ad/ckpts'),

    ('gn05_sng',    'sng',    0.05, None, 'test_sp_ad/ckpts'),
    ('gn05_single', 'single', 0.05, None, 'test_sp_ad/ckpts'),
    ('gn05_multi',  '',       0.05, None, 'test_sp_ad/ckpts'),
    ('gn05_ad',     'ad',     0.05, None, 'test_sp_ad/ckpts'),

    ('ln01_sng',    'sng',    0.084, low_pass, 'test_sp_ad/ckpts'),
    ('ln01_single', 'single', 0.084, low_pass, 'test_sp_ad/ckpts'),
    ('ln01_multi',  '',       0.084, low_pass, 'test_sp_ad/ckpts'),
    ('ln01_ad',     'ad',     0.084, low_pass, 'test_sp_ad/ckpts'),

    ('ln05_sng',    'sng',    0.42, low_pass, 'test_sp_ad/ckpts'),
    ('ln05_single', 'single', 0.42, low_pass, 'test_sp_ad/ckpts'),
    ('ln05_multi',  '',       0.42, low_pass, 'test_sp_ad/ckpts'),
    ('ln05_ad',     'ad',     0.42, low_pass, 'test_sp_ad/ckpts'),
]


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
figure_matrix = [5, 4]
figure_size = (16, 20)
num_axes = reduce(lambda x, y: x * y, figure_matrix)
validation_set = TPZDataset('./data/gdgn_cir3_e64_64_c8_validate/', 2000, device=device, tqdm=True)
bce_loss = BCELoss(reduction='none')

REPEAT = 1
save_dir = 'test_sp_ad/figures/'
use_noise_filter = True

os.makedirs(save_dir, exist_ok=True)

def validate(model: Module,
             loader,
             loss_fn: Callable[[Tensor, Tensor], Tensor],
             noise_coef: float,
             noise_filter: Optional[Module]=None,
             repeat: int=1) -> Tensor:
    model.eval()
    loss = []

    for _ in range(repeat):
        for x, label in tqdm(loader, desc='Validation', unit='batch'):
            x = x.clone()
            label = label.to(dtype=torch.float32)
            noise = torch.randn_like(x[:, :, 0, :]) * noise_coef
            if noise_filter:
                noise = noise_filter(noise)
            noise = x[:, :, 0, :] * noise
            x[:, :, 0, :] += noise
            y_pred = model(x)
            no_reduct = loss_fn(y_pred.squeeze(1), label).detach().cpu().mean(0)
            no_reduct.squeeze_(0) # shape: (h, w)
            loss.append(no_reduct)

    return torch.stack(loss, dim=0).mean(0)


### Validation and Visualization Scripts ###

from matplotlib import pyplot as plt

fig = plt.figure('', figsize=figure_size)
model_cursor = 0

for tag, type_, noise_coef, noise_filter, ckpts_path in settings:
    model, name = build_model(device, tag, type_, ckpts_path)
    model.eval()
    model_cursor += 1 # starts from 1

    cross_entropy_loss = validate(model, validation_set.loader(500), bce_loss,
                                  noise_coef, noise_filter, repeat=REPEAT)

    axes = fig.add_subplot(*figure_matrix, model_cursor)
    axes.imshow(cross_entropy_loss, cmap='rainbow', vmin=0, vmax=0.2)
    axes.invert_yaxis()
    axes.set_title(name)

fig.suptitle(f'Validation loss distribution (cir3)')
fig.savefig(os.path.join(save_dir, 'vld_cir3.png'))
