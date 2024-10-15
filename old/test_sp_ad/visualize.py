
import os
import sys
from typing import Callable, Dict, Optional
from functools import reduce

sys.path.append('./src')

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.figure import Figure

from common import loss_fn as cross_entropy
from fractional import Fractional
from unet_100 import build_model
from dataset import NPZDataset


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
low_pass = Fractional(252, device=device)
low_pass.from_npz(r"./data/laplace_beltrami_63_63.npz")
low_pass.initialize(s=-0.75)
low_pass.s.requires_grad_(False)

settings = [
#   ('tag',       'type', 'noise', 'filter', 'ckpts_path')
    ('nn_sng',      'sng',    0.0,  None, 'test_no_noise/ckpts'),
    ('gn01_sng',    'sng',    0.01, None, 'test_sp_ad/ckpts'),
    ('gn05_sng',    'sng',    0.05, None, 'test_sp_ad/ckpts'),
    ('ln01_sng',    'sng',    0.084, low_pass, 'test_sp_ad/ckpts'),
    ('ln05_sng',    'sng',    0.42, low_pass, 'test_sp_ad/ckpts'),

    ('nn_single',   'single', 0.0,  None, 'test_no_noise/ckpts'),
    ('gn01_single', 'single', 0.01, None, 'test_sp_ad/ckpts'),
    ('gn05_single', 'single', 0.05, None, 'test_sp_ad/ckpts'),
    ('ln01_single', 'single', 0.084, low_pass, 'test_sp_ad/ckpts'),
    ('ln05_single', 'single', 0.42, low_pass, 'test_sp_ad/ckpts'),

    ('nn_multi',    '',       0.0,  None, 'test_no_noise/ckpts'),
    ('gn01_multi',  '',       0.01, None, 'test_sp_ad/ckpts'),
    ('gn05_multi',  '',       0.05, None, 'test_sp_ad/ckpts'),
    ('ln01_multi',  '',       0.084, low_pass, 'test_sp_ad/ckpts'),
    ('ln05_multi',  '',       0.42, low_pass, 'test_sp_ad/ckpts'),
]

figure_matrix = [3, 5]
row_title = ['s=0', 'Single', 'Multi']
col_title = ['No Noise', 'Gaussian 1%', 'Gaussian 5%', 'Low-freq 1%', 'Low-freq 5%']
figure_size = (20, 12)
num_axes = reduce(lambda x, y: x * y, figure_matrix)
validation_set = NPZDataset('./data/gdgn_cir3_e64_64_c8_validate/', 2000,
                            channel_keys=['1', '2', '3', '4', '5', '6', '8', '16'])

REPEAT = 1
NO_PLOT = False
save_dir = 'test_sp_ad/figures/'
use_noise_filter = True
os.makedirs(save_dir, exist_ok=True)

### Validation and Visualization Scripts ###

ID = [110, 120, 130, 140]
model_cursor = 0
figs = {plt.figure(f"Data{i}", figsize=figure_size) for i in ID}
gss = {figs[i].add_gridspec(figure_matrix[0]+1, figure_matrix[1]+1) for i in ID}

x = np.linspace(-1, 1, 64)
y = np.linspace(-1, 1, 64)
X, Y = np.meshgrid(x, y, indexing='ij')

for tag, type_, noise_coef, noise_filter, ckpts_path in settings:
    model, name = build_model(device, tag, type_, ckpts_path)
    model.eval()
    model_cursor += 1 # starts from 1

    # for each sample
    for i in ID:
        fig = figs[i]
        gs = gss[i]
        fig.add_subplot(gs[0, 0])
        data = validation_set[i][0].clone().to(device)

        noise = torch.randn_like(data[..., 0, :]) * noise_coef
        if noise_filter:
            noise = noise_filter(noise)
        noise = data[..., 0, :] * noise
        data[..., 0, :] += noise
        pred = model(data[None, ...])
        axes = fig.add_subplot(figure_matrix[0], figure_matrix[1], model_cursor)
        axes.pcolormesh(X, Y, pred.detach().cpu().reshape(64, 64), cmap='jet', vmin=0, vmax=1)

        file_ = np.load(f'data/gdgn_cir3_e64_64_c8_validate/{i}.npz')
        ctrs, rads = file_['ctrs'], file_['rads']
        for j in range(ctrs.shape[0]):
            circle = Circle((ctrs[j, 0], ctrs[j, 1]), rads[j], color='white', fill=False, linewidth=1.5, linestyle='--')
            axes.add_patch(circle)
        axes.invert_yaxis()
        axes.set_title(name)

if not NO_PLOT:
    for i in ID:
        figs[i].tight_layout()
        # figs[i].suptitle(f'validate - (cir3)Data{i}', fontsize=18)
        figs[i].savefig(f'{save_dir}vis_cir3_{i}.png')
