"""
绘制无噪声时的预测图形表格，包含各个样本（行）、各模型（列）下的预测结果。
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
from unet_100 import build_model
from dataset import NPZDataset, NPYDataset


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

settings = [
#   ('pos', 'tag',     'type', 'sample',   'filter', 'ckpts_path')
    (6,  'nn_sng',     'sng',    715,  None, 'test_fem_version/ckpts'),
    (7, 'nn_single',   'single', 715,  None, 'test_fem_version/ckpts'),
    (8, 'nn_multi',    'multi',  715,  None, 'test_fem_version/ckpts'),

    (10, 'nn_sng',      'sng',    1102,  None, 'test_fem_version/ckpts'),
    (11, 'nn_single',   'single', 1102,  None, 'test_fem_version/ckpts'),
    (12, 'nn_multi',    'multi',  1102,  None, 'test_fem_version/ckpts'),

    (14, 'nn_sng',      'sng',    728,  None, 'test_fem_version/ckpts'),
    (15, 'nn_single',   'single', 728,  None, 'test_fem_version/ckpts'),
    (16, 'nn_multi',    'multi',  728,  None, 'test_fem_version/ckpts'),

    (18, 'nn_sng',      'sng',    306,  None, 'test_fem_version/ckpts'),
    (19, 'nn_single',   'single', 306,  None, 'test_fem_version/ckpts'),
    (20, 'nn_multi',    'multi',  306,  None, 'test_fem_version/ckpts'),
]

titles = [
#   ('pos', 'title', 'rotation')
    (2,  '$\gamma=0$',    0),
    (3,  'Single',        0),
    (4,  'Multi',         0),
    (5,  'sample 1',     90),
    (9,  'sample 2',     90),
    (13, 'sample 3',     90),
    (17, 'sample 4',     90)
]

figure_matrix = [5, 4]
wr = [0.15, 1, 1, 1]
hr = [0.15, 1, 1, 1, 1]
figure_size = (12, 15)
num_axes = reduce(lambda x, y: x * y, figure_matrix)
gd_set = NPYDataset("data/cir3_e64_64_c8/gd", [str(i) for i in range(1200)])
gn = torch.from_numpy(np.load('data/cir3_e64_64_c8/gn.npy')).to(device)
label_set = NPZDataset("data/cir3_e64_64_c8/inclusion", [str(i) for i in range(1200)])


save_dir = 'test_fem_version/figures/'
os.makedirs(save_dir, exist_ok=True)

### Validation and Visualization Scripts ###

fig = plt.figure(f"NoNoise", figsize=figure_size)
gs = GridSpec(figure_matrix[0], figure_matrix[1],
              width_ratios=wr, height_ratios=hr)

x = np.linspace(-1, 1, 64)
y = np.linspace(-1, 1, 64)

X, Y = np.meshgrid(x, y, indexing='ij')

for pos, tag, type_, sample_id, noise_filter, ckpts_path in settings:
    model, name = build_model(device, tag, type_, ckpts_path)
    model.eval()
    pos_col = (pos-1) % figure_matrix[1]
    pos_row = (pos-1) // figure_matrix[1]

    gd = gd_set[sample_id].to(device)
    data = torch.stack([gd, gn], dim=-2)
    label = label_set[sample_id][-1].to(device)

    pred = model(data[None, ...])
    label = label.reshape(pred.shape).to(dtype=pred.dtype)
    loss = cross_entropy(pred, label).item()

    axes = fig.add_subplot(gs[pos_row, pos_col])
    axes.pcolormesh(X, Y, pred.detach().cpu().reshape(64, 64), cmap='jet', vmin=0, vmax=1)

    file_ = np.load(f'data/cir3_e64_64_c8/inclusion/{sample_id}.npz')
    ctrs, rads = file_['ctrs'], file_['rads']

    for j in range(ctrs.shape[0]):
        circle = Circle((ctrs[j, 0], ctrs[j, 1]), rads[j], color='white', fill=False, linewidth=1.5, linestyle='--')
        axes.add_patch(circle)

    axes.invert_yaxis()
    axes.set_title(f"loss={loss:.4e}")

for pos, title, rot in titles:
    pos_col = (pos-1) % figure_matrix[1]
    pos_row = (pos-1) // figure_matrix[1]
    axes = fig.add_subplot(gs[pos_row, pos_col])
    axes.text(0.5, 0.5, title, ha='center', va='center', fontsize=16, rotation=rot)
    axes.axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'vis_cir3_nonoise.png'))
