"""
绘制单个样本的预测剖面图。
"""
import os
from typing import List

import numpy as np
import torch
# from torch.nn.functional import binary_cross_entropy_with_logits
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from lafemeit.model import build_eit_model, Fractional
from lafemeit.utils import NPZDataset, NPYDataset


torch.manual_seed(-26)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
low_pass = Fractional(252, device=device)
low_pass.from_npz("lafem/data/laplace_beltrami_63_63.npz")
low_pass.initialize(gamma=-0.75)
low_pass.gamma.requires_grad_(False)

settings = [
#   ('linestyle', 'tag',       'type', 'noise', 'filter', 'ckpts_path')
    ('r',  'nn_nograd',      'nograd',    0.0,  None, 'lafem/ckpts'),
    ('r--', 'ln01_nograd',    'nograd',    0.0, low_pass, 'lafem/ckpts'),
    ('r:', 'ln05_nograd',    'nograd',    0.0, low_pass, 'lafem/ckpts'),

    ('g', 'nn_single',   'single', 0.0,  None, 'lafem/ckpts'),
    ('g--', 'ln01_single', 'single', 0.0, low_pass, 'lafem/ckpts'),
    ('g:', 'ln05_single', 'single', 0.0, low_pass, 'lafem/ckpts'),

    ('b', 'nn_multi',    'multi',  0.0,  None, 'lafem/ckpts'),
    ('b--', 'ln01_multi',  'multi',  0.0, low_pass, 'lafem/ckpts'),
    ('b:', 'ln05_multi',  'multi',  0.0, low_pass, 'lafem/ckpts'),
]


LABEL_POS = (4, 1)

figure_matrix = [5, 4]
figure_size = (9, 12)

gd_set1 = NPYDataset("lafem/data/cir5_e64_64_c8/gd", [str(i) for i in range(0, 200)])
gn1 = torch.from_numpy(np.load('lafem/data/cir5_e64_64_c8/gn.npy')).to(device)
label_set1 = NPZDataset("lafem/data/cir5_e64_64_c8/inclusion", [str(i) for i in range(0, 200)])

gd_set2 = NPYDataset("lafem/data/cir5_e64_64_c8_modified/gd", [str(i) for i in range(0, 200)])
gn2 = torch.from_numpy(np.load('lafem/data/cir5_e64_64_c8_modified/gn.npy')).to(device)
label_set2 = NPZDataset("lafem/data/cir5_e64_64_c8_modified/inclusion", [str(i) for i in range(0, 200)])


samples = [
    (gd_set1, gn1, label_set1, 1, 51),
    (gd_set2, gn2, label_set2, 105, 26),
    (gd_set1, gn1, label_set1, 113, 32),
]

save_dir = 'lafem/figure/'
use_noise_filter = True
os.makedirs(save_dir, exist_ok=True)

### Validation and Visualization Scripts ###
# 0, 25, 30, 43, 44, 53, 67, 83, 88, 94, 96

x = np.linspace(-1, 1, 64)
y = np.linspace(-1, 1, 64)
X, Y = np.meshgrid(x, y, indexing='ij')
NOISE = torch.randn((20, 8, 252), dtype=torch.float64) # (N, channel, dofs)

axes_list: List[Axes] = []
N = len(samples)
fig = plt.figure(figsize=figure_size)

for i in range(N):
    axes_list.append(fig.add_subplot(N, 1, i + 1))


for linestyle, tag, type_, noise_coef, noise_filter, ckpts_path in settings:
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
    for idx, (gd_set, gn, _, i, yidx) in enumerate(samples):
        gd = gd_set[i].to(device)
        data = torch.stack([gd, gn], dim=1)[None, ...]
        pred = model(data).detach()
        pred = pred.sigmoid_().squeeze(0).numpy().reshape(64, 64)[:, yidx]

        axes = axes_list[idx]
        axes.plot(x, pred, linestyle, label=f'{tag}', linewidth=1)

for idx, (_, _, label_set, i, yidx) in enumerate(samples):
    label = label_set[i][0].to(torch.float32).numpy().reshape(64, 64)[:, yidx]
    axes = axes_list[idx]
    axes.plot(x, label, "black", label='label', linewidth=1)
    axes.set_title(f"y={y[yidx]:.2f}")
    axes.legend()


fig.tight_layout()
fig.savefig(os.path.join(save_dir, f'sectional_cir5_lowfreq.png'))
plt.show()