
import os
import sys
from typing import Callable, Dict, Optional
from functools import reduce

sys.path.append('./src')

import torch
from torch import Tensor
from torch.nn import Module, BCELoss
from torchvision.transforms import CenterCrop
from tqdm import tqdm

from common import loss_fn as cross_entropy
from fractional import Fractional
from unet_100 import build_model
from dataset import TPZDataset


low_pass = Fractional(252, device='cpu')
low_pass.from_npz(r"./data/laplace_beltrami_63_63.npz")
low_pass.initialize(s=-0.75)
low_pass.s.requires_grad_(False)

MODEL_A = ('gn01_sng',    'sng',    0.00, None, 'test_sp_ad/ckpts')
MODEL_B = ('gn01_single', 'single', 0.00, None, 'test_sp_ad/ckpts')

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
figure_matrix = [6, 4]
figure_size = (16, 24)
num_axes = reduce(lambda x, y: x * y, figure_matrix)
validation_set = TPZDataset('./data/gdgn_cir2_e64_64_c8_validate/', 2000, device=device, tqdm=True)
loss_fn = BCELoss(reduction='mean')

REPEAT = 1
save_dir = 'test_sp_ad/figures/'
use_noise_filter = True


def validate(model: Module,
             loader,
             loss_fn: Callable[[Tensor, Tensor], Tensor],
             noise_coef: float,
             noise_filter: Optional[Module]=None,
             transform: Optional[Callable[[Tensor], Tensor]]=None) -> Tensor:
    model.eval()
    loss = []

    for x, label in tqdm(loader, desc='Validation', unit='batch'):
        x = x.unsqueeze(0).clone()
        label = label.to(dtype=torch.float32)
        label = transform(label) if transform else label
        noise = torch.randn_like(x[:, :, 0, :]) * noise_coef
        if noise_filter:
            noise = noise_filter(noise)
        noise = x[:, :, 0, :] * noise
        x[:, :, 0, :] += noise
        y_pred = model(x).squeeze(0, 1)
        y_pred = transform(y_pred) if transform else y_pred
        non_reducted = loss_fn(y_pred, label).detach().cpu().item()
        loss.append(non_reducted)

    loss_vec = torch.tensor(loss)
    return loss_vec


### Validation and Visualization Scripts ###

result_string = ""
result_rounded = ""
result = []

for tag, type_, noise_coef, noise_filter, ckpts_path in (MODEL_A, MODEL_B):
    model, name = build_model(device, tag, type_, ckpts_path)
    model.eval()

    cross_entropy_loss = validate(model, validation_set, cross_entropy, noise_coef, noise_filter,
                                  transform=CenterCrop(32))
    print(f'Validation loss for {name}: {cross_entropy_loss}')
    result.append(cross_entropy_loss)

print(torch.sort(result[1] - result[0]).indices.tolist())
