"""
绘制单个样本的预测图形表格，包含各个模型（行）、各水平噪声（列）下的预测结果。
"""
import os
from functools import reduce

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec

from lafemeit.model import build_eit_model, Fractional
from lafemeit.utils import NPZDataset, NPYDataset


torch.manual_seed(-26)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
low_pass = Fractional(252, device=device)
low_pass.from_npz("lafem/data/laplace_beltrami_63_63.npz")
low_pass.initialize(gamma=-0.75)
low_pass.gamma.requires_grad_(False)

settings = [
#   ('row', 'col', 'tag',       'type', 'noise', 'filter', 'ckpts_path')
    (1, 1,  'nn_nograd',      'nograd',    0.0,  None, 'lafem/ckpts'),
    (1, 2, 'ln01_nograd',    'nograd',    0.084, low_pass, 'lafem/ckpts'),
    (1, 3, 'ln05_nograd',    'nograd',    0.42, low_pass, 'lafem/ckpts'),

    (2, 1, 'nn_single',   'single', 0.0,  None, 'lafem/ckpts'),
    (2, 2, 'ln01_single', 'single', 0.084, low_pass, 'lafem/ckpts'),
    (2, 3, 'ln05_single', 'single', 0.42, low_pass, 'lafem/ckpts'),

    (3, 1, 'nn_multi',    'multi',  0.0,  None, 'lafem/ckpts'),
    (3, 2, 'ln01_multi',  'multi',  0.084, low_pass, 'lafem/ckpts'),
    (3, 3, 'ln05_multi',  'multi',  0.42, low_pass, 'lafem/ckpts'),
]

titles = [
#   ('row', 'col', 'title', 'rotation')
    (0, 1,  'No Noise',        0),
    (0, 2,  '1% Noise',     0),
    (0, 3,  '5% Noise',     0),
    (1, 0,  '$\gamma = 0$',    90),
    (2, 0, 'Single',          90),
    (3, 0, 'Multi',           90)
]

LABEL_POS = (4, 1)

figure_matrix = [5, 4]
wr = [0.15, 1, 1, 1]
hr = [0.15, 1, 1, 1, 1]
figure_size = (12, 16)
num_axes = reduce(lambda x, y: x * y, figure_matrix)
gd_set = NPYDataset("lafem/data/cir5_e64_64_c8_modified/gd", [str(i) for i in range(12000)])
gn = torch.from_numpy(np.load('lafem/data/cir5_e64_64_c8_modified/gn.npy')).to(device)
label_set = NPZDataset("lafem/data/cir5_e64_64_c8_modified/inclusion", [str(i) for i in range(12000)])


save_dir = 'lafem/figure/'
use_noise_filter = True
os.makedirs(save_dir, exist_ok=True)

### Validation and Visualization Scripts ###

ID = [105,]
figs = {i: plt.figure(f"Data{i}", figsize=figure_size) for i in ID}
gs = GridSpec(figure_matrix[0], figure_matrix[1],
              width_ratios=wr, height_ratios=hr)

x = np.linspace(-1, 1, 64)
y = np.linspace(-1, 1, 64)
X, Y = np.meshgrid(x, y, indexing='ij')
NOISE = torch.randn((20, 8, 252), dtype=torch.float64) # (N, channel, dofs)

for pos_row, pos_col, tag, type_, noise_coef, noise_filter, ckpts_path in settings:
    model, name = build_eit_model(
        "unet100", 63, 8,
        tag=tag,
        fractype=type_,
        eigen_file="lafem/data/laplace_beltrami_63_63.npz",
        ckpts_path=ckpts_path,
        device=device
    )
    model.eval()

    # for each sample/figure
    for i in ID:
        if noise_filter:
            noise_ = noise_filter(NOISE * noise_coef)
        else:
            noise_ = NOISE * noise_coef

        fig = figs[i]
        gd = gd_set[i].to(device)
        data = torch.empty([20, 8, 2, 252], dtype=gd.dtype, device=gd.device) # new memory
        torch.multiply(noise_, gd[None, ...], out=data[:, :, 0, :])
        data[:, :, 0, :] += gd[None, ...]
        data[:, :, 1, :] = gn[None, ...]
        label = label_set[i][-1].to(device)

        pred = model(data).mean(0)
        label = label.reshape(pred.shape).to(dtype=pred.dtype)
        loss = binary_cross_entropy(pred, label).item()

        axes = fig.add_subplot(gs[pos_row, pos_col])
        axes.pcolormesh(X, Y, pred.detach().cpu().reshape(64, 64), cmap='jet', vmin=0, vmax=1)

        file_ = np.load(f'lafem/data/cir5_e64_64_c8_modified/inclusion/{i}.npz')
        ctrs, rads = file_['ctrs'], file_['rads']

        for j in range(ctrs.shape[0]):
            circle = Circle((ctrs[j, 0], ctrs[j, 1]), rads[j], color='white', fill=False, linewidth=1.25, linestyle='--')
            axes.add_patch(circle)

        axes.set_title(f"loss={loss:.4e}")

for i in ID:
    for pos_row, pos_col, title, rot in titles:
        fig = figs[i]
        axes = fig.add_subplot(gs[pos_row, pos_col])
        axes.text(0.5, 0.5, title, ha='center', va='center', fontsize=16, rotation=rot)
        axes.axis('off')

        label = label_set[i][-1]
        axes = fig.add_subplot(gs[*LABEL_POS])
        axes.pcolormesh(X, Y, label.reshape(64, 64), cmap='jet', vmin=0, vmax=1)
        axes.set_title('Inclusion')

    figs[i].tight_layout()
    figs[i].savefig(os.path.join(save_dir, f'vis_cir5_{i}_lowfreq_m.png'))
