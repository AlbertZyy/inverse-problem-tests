
import os
import sys
from typing import Callable, Dict, Optional
from functools import reduce

sys.path.append('./src')

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torchvision.transforms import CenterCrop
from tqdm import tqdm

from common import loss_fn as cross_entropy
from fractional import Fractional
from unet_100 import build_model
from dataset import TPZDataset


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
figure_size = (20, 12)
num_axes = reduce(lambda x, y: x * y, figure_matrix)
validation_set = TPZDataset('./data/gdgn_cir3_e64_64_c8_validate/', 2000,
                            channel_keys=['1', '2', '3', '4', '5', '6', '8', '16'],
                            device=device,
                            tqdm=True)

REPEAT = 1
NO_EVALUATE = True
NO_PLOT = False
save_dir = 'test_sp_ad/figures/'
use_noise_filter = True


if not NO_PLOT:
    os.makedirs(save_dir, exist_ok=True)

def validate(model: Module,
             loader,
             loss_fn: Callable[[Tensor, Tensor], Tensor],
             noise_coef: float,
             noise_filter: Optional[Module]=None,
             transform: Optional[Callable[[Tensor], Tensor]]=None,
             repeat: int=1):
    model.eval()
    loss = 0
    count = 0

    for _ in range(repeat):
        for x, label in tqdm(loader, desc='Validation', unit='batch'):
            x = x.clone()
            label = label.to(dtype=torch.float32)
            label = transform(label) if transform else label
            noise = torch.randn_like(x[:, :, 0, :]) * noise_coef
            if noise_filter:
                noise = noise_filter(noise)
            noise = x[:, :, 0, :] * noise
            x[:, :, 0, :] += noise
            y_pred = model(x).squeeze(1)
            y_pred = transform(y_pred) if transform else y_pred
            loss += loss_fn(y_pred, label).detach().cpu().item()
            count += 1

    return loss / count


### Validation and Visualization Scripts ###

model_cursor = 0

if not NO_PLOT:
    from matplotlib.figure import Figure
    figs: Dict[int, Figure] = {}
    ID = [110, 120, 130, 140]

result_string = ""
result_rounded = ""
x = np.linspace(-1, 1, 64)
y = np.linspace(-1, 1, 64)
X, Y = np.meshgrid(x, y, indexing='ij')

for tag, type_, noise_coef, noise_filter, ckpts_path in settings:
    model, name = build_model(device, tag, type_, ckpts_path)
    model.eval()
    model_cursor += 1 # starts from 1

    if not NO_EVALUATE:
        cross_entropy_loss = validate(
            model, validation_set.loader(500), cross_entropy, noise_coef,
            noise_filter, transform=CenterCrop(32), repeat=REPEAT)
        print(f'Validation loss for {name}: {cross_entropy_loss}')
        result_string += f"{cross_entropy_loss}\n"
        result_rounded += f"{round(cross_entropy_loss, 5)}\n"

    if not NO_PLOT:
        from matplotlib import pyplot as plt
        from matplotlib.patches import Circle

        # for each sample
        for i in ID:
            fig = figs.get(i, None) or plt.figure(f"Data{i}", figsize=figure_size)
            fig.tight_layout()
            figs[i] = fig
            data = validation_set[i][0].clone()

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
                circle = Circle((ctrs[j, 0], ctrs[j, 1]), rads[j], color='black', fill=False, linewidth=1.5)
                axes.add_patch(circle)
            axes.invert_yaxis()
            axes.set_title(name)

if not NO_EVALUATE:
    with open(os.path.join(save_dir, 'result.txt'), 'w') as f:
        f.write(result_string)
        f.write('\n')
        f.write(result_rounded)

if not NO_PLOT:
    for i in ID:
        # figs[i].suptitle(f'validate - (cir3)Data{i}', fontsize=18)
        figs[i].savefig(f'{save_dir}vis_cir3_{i}.png')
