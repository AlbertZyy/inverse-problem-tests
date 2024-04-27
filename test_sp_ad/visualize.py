
import os
import sys
from typing import Callable, Dict, Optional
from functools import reduce

sys.path.append('./src')

import torch
from torch import Tensor
from torch.nn import MSELoss, Module
from tqdm import tqdm

from common import loss_fn as cross_entropy
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
figure_matrix = [6, 4]
figure_size = (16, 24)
num_axes = reduce(lambda x, y: x * y, figure_matrix)
validation_set = TPZDataset('./data/gdgn_64_64_validate/', 200, device=device)

REPEAT = 10
NO_EVALUATE = True
NO_PLOT = False
save_dir = 'test_sp_ad/figures/'
use_noise_filter = True


if not NO_PLOT:
    os.makedirs(save_dir, exist_ok=True)


mse = MSELoss()

def validate(model: Module,
             loader,
             loss_fn: Callable[[Tensor, Tensor], Tensor],
             noise_coef: float,
             noise_filter: Optional[Module]=None,
             repeat: int=1):
    model.eval()
    loss = 0
    count = 0

    for _ in range(repeat):
        for x, label in tqdm(loader, desc='Validation', unit='batch'):
            x = x.clone()
            noise = torch.randn_like(x[:, :, 0, :]) * noise_coef
            if noise_filter:
                noise = noise_filter(noise)
            noise = x[:, :, 0, :] * noise
            x[:, :, 0, :] += noise
            y_pred = model(x)
            loss += loss_fn(y_pred, label.flatten().to(dtype=torch.float32)).detach().cpu().item()
            count += 1

    return loss / count


### Validation and Visualization Scripts ###

model_cursor = 0

if not NO_PLOT:
    from matplotlib.figure import Figure
    figs: Dict[int, Figure] = {}
    ID = [67, ]

result_string = ""
result_rounded = ""

for tag, type_, noise_coef, noise_filter, ckpts_path in settings:
    model, name = build_model(device, tag, type_, ckpts_path)
    model.eval()
    model_cursor += 1 # starts from 1

    if not NO_EVALUATE:
        cross_entropy_loss = validate(model, validation_set.loader(200), cross_entropy, noise_coef, noise_filter, repeat=REPEAT)
        # mse_loss = validate(model, validation_set.loader(200), mse, noise_coef, noise_filter)
        print(f'Validation loss for {name}: {cross_entropy_loss}')
        # print(f"  - mse loss: {mse_loss}")
        result_string += f"{cross_entropy_loss}\n"
        result_rounded += f"{round(cross_entropy_loss, 5)}\n"

    if not NO_PLOT:
        from matplotlib import pyplot as plt

        # for each sample
        for i in ID:
            fig = figs.get(i, None) or plt.figure(f"Data{i}", figsize=figure_size)
            figs[i] = fig
            data = validation_set[i][0].clone()

            noise = torch.randn_like(data[..., 0, :]) * noise_coef
            if noise_filter:
                noise = noise_filter(noise)
            noise = data[..., 0, :] * noise
            data[..., 0, :] += noise

            pred = model(data[None, ...])
            axes = fig.add_subplot(*figure_matrix, model_cursor)
            axes.imshow(pred.detach().reshape(64, 64))
            axes.invert_yaxis()
            axes.set_title(name)

if not NO_EVALUATE:
    with open(os.path.join(save_dir, 'result.txt'), 'w') as f:
        f.write(result_string)
        f.write('\n')
        f.write(result_rounded)

if not NO_PLOT:
    for i in ID:
        data, label = validation_set[i]
        axes = figs[i].add_subplot(*figure_matrix, num_axes)
        axes.imshow(label.to(dtype=torch.float32))
        axes.invert_yaxis()
        axes.set_title('label')
        figs[i].suptitle(f'validate - (cir3)Data{i}')
        figs[i].savefig(f'{save_dir}vis_cir3_{i}.png')
