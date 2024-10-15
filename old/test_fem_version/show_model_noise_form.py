"""
绘制单个样本的预测图形表格，包含各个模型（行）、各水平噪声（列）下的预测结果。
"""

import os
import sys
from functools import reduce

sys.path.append('./src')

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec

from common import loss_fn as cross_entropy
from fractional import Fractional
from unet_100 import build_model
from dataset import NPZDataset, NPYDataset


torch.manual_seed(-26)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
low_pass = Fractional(252, device=device)
low_pass.from_npz(r"./data/laplace_beltrami_63_63_torch.npz")
low_pass.initialize(s=-0.75)
low_pass.s.requires_grad_(False)

settings = [
#   ('pos', 'tag',       'type', 'noise', 'filter', 'ckpts_path')
    (8,  'nn_sng',      'sng',    0.0,  None, 'test_fem_version/ckpts'),
    (9,  'gn01_sng',    'sng',    0.01, None, 'test_fem_version/ckpts'),
    (10, 'gn05_sng',    'sng',    0.05, None, 'test_fem_version/ckpts'),
    (11, 'ln01_sng',    'sng',    0.084, low_pass, 'test_fem_version/ckpts'),
    (12, 'ln05_sng',    'sng',    0.42, low_pass, 'test_fem_version/ckpts'),

    (14, 'nn_single',   'single', 0.0,  None, 'test_fem_version/ckpts'),
    (15, 'gn01_single', 'single', 0.01, None, 'test_fem_version/ckpts'),
    (16, 'gn05_single', 'single', 0.05, None, 'test_fem_version/ckpts'),
    (17, 'ln01_single', 'single', 0.084, low_pass, 'test_fem_version/ckpts'),
    (18, 'ln05_single', 'single', 0.42, low_pass, 'test_fem_version/ckpts'),

    (20, 'nn_multi',    'multi',  0.0,  None, 'test_fem_version/ckpts'),
    (21, 'gn01_multi',  'multi',  0.01, None, 'test_fem_version/ckpts'),
    (22, 'gn05_multi',  'multi',  0.05, None, 'test_fem_version/ckpts'),
    (23, 'ln01_multi',  'multi',  0.084, low_pass, 'test_fem_version/ckpts'),
    (24, 'ln05_multi',  'multi',  0.42, low_pass, 'test_fem_version/ckpts'),
]

titles = [
#   ('pos', 'title', 'rotation')
    (2,  'No Noise',        0),
    (3,  'Gaussian 1%',     0),
    (4,  'Gaussian 5%',     0),
    (5,  'Low-freq 1%',     0),
    (6,  'Low-freq 5%',     0),
    (7,  '$\gamma = 0$',    90),
    (13, 'Single',          90),
    (19, 'Multi',           90)
]

figure_matrix = [4, 6]
wr = [0.15, 1, 1, 1, 1, 1]
hr = [0.15, 1, 1, 1]
figure_size = (20, 12)
num_axes = reduce(lambda x, y: x * y, figure_matrix)
gd_set = NPYDataset("data/cir3_e64_64_c8/gd", [str(i) for i in range(1000)])
gn = torch.from_numpy(np.load('data/cir3_e64_64_c8/gn.npy')).to(device)
label_set = NPZDataset("data/cir3_e64_64_c8/inclusion", [str(i) for i in range(1000)])


save_dir = 'test_fem_version/figures/'
use_noise_filter = True
os.makedirs(save_dir, exist_ok=True)

### Validation and Visualization Scripts ###

ID = [122, 222, 621, 928]
figs = {i: plt.figure(f"Data{i}", figsize=figure_size) for i in ID}
gs = GridSpec(figure_matrix[0], figure_matrix[1],
              width_ratios=wr, height_ratios=hr)

x = np.linspace(-1, 1, 64)
y = np.linspace(-1, 1, 64)
X, Y = np.meshgrid(x, y, indexing='ij')
NOISE = torch.randn((8, 252), dtype=torch.float64)

for pos, tag, type_, noise_coef, noise_filter, ckpts_path in settings:
    model, name = build_model(device, tag, type_, ckpts_path)
    model.eval()

    pos_col = (pos-1) % figure_matrix[1]
    pos_row = (pos-1) // figure_matrix[1]

    # for each sample/figure
    for i in ID:
        fig = figs[i]
        gd = gd_set[i].to(device)
        data = torch.stack([gd, gn], dim=-2) # new memory
        label = label_set[i][-1].to(device)

        if noise_filter:
            noise_ = noise_filter(NOISE * noise_coef)
        else:
            noise_ = NOISE * noise_coef
        data[:, 0, :] += data[:, 0, :] * noise_

        pred = model(data[None, ...])
        label = label.reshape(pred.shape).to(dtype=pred.dtype)
        loss = cross_entropy(pred, label).item()

        axes = fig.add_subplot(gs[pos_row, pos_col])
        axes.pcolormesh(X, Y, pred.detach().cpu().reshape(64, 64), cmap='jet', vmin=0, vmax=1)

        file_ = np.load(f'data/cir3_e64_64_c8/inclusion/{i}.npz')
        ctrs, rads = file_['ctrs'], file_['rads']

        for j in range(ctrs.shape[0]):
            circle = Circle((ctrs[j, 0], ctrs[j, 1]), rads[j], color='white', fill=False, linewidth=1.5, linestyle='--')
            axes.add_patch(circle)

        axes.invert_yaxis()
        axes.set_title(f"loss={loss:.4e}")

for i in ID:
    for pos, title, rot in titles:
        fig = figs[i]
        pos_col = (pos-1) % figure_matrix[1]
        pos_row = (pos-1) // figure_matrix[1]
        axes = fig.add_subplot(gs[pos_row, pos_col])
        axes.text(0.5, 0.5, title, ha='center', va='center', fontsize=16, rotation=rot)
        axes.axis('off')

    figs[i].tight_layout()
    figs[i].savefig(os.path.join(save_dir, f'vis_cir3_{i}.png'))
