
import sys
from typing import Callable, Optional

sys.path.append('./src')

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.utils.data import StackDataset, DataLoader
from torchvision.transforms import CenterCrop
from tqdm import tqdm
from lafemeit.model import build_eit_model, Fractional
from lafemeit.utils import NPZDataset, NPYDataset


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
low_pass = Fractional(252, device=device)
low_pass.from_npz(r"./lafem/data/laplace_beltrami_63_63.npz")
low_pass.initialize(gamma=-0.75)
low_pass.gamma.requires_grad_(False)


settings = [
#   ('tag',       'type', 'noise', 'filter', 'ckpts_path')
    # ('nn_nograd',      'nograd',    0.0,  None, 'lafem/ckpts'),
    # ('ln01_nograd',    'nograd',    0.084, low_pass, 'lafem/ckpts'),
    # ('ln05_nograd',    'nograd',    0.42, low_pass, 'lafem/ckpts'),

    ('nn_single',   'single', 0.0,  None, 'lafem/ckpts'),
    # ('ln01_single', 'single', 0.084, low_pass, 'lafem/ckpts'),
    # ('ln05_single', 'single', 0.42, low_pass, 'lafem/ckpts'),

    # ('nn_multi',    'multi',       0.0,  None, 'lafem/ckpts'),
    # ('ln01_multi',  'multi',       0.084, low_pass, 'lafem/ckpts'),
    # ('ln05_multi',  'multi',       0.42, low_pass, 'lafem/ckpts'),
]

gd_set = NPYDataset("lafem/data/cir3_e64_64_c8/gd", [str(i) for i in range(10000, 10200)])
gn = torch.from_numpy(np.load('lafem/data/cir3_e64_64_c8/gn.npy')).to(device)
label_set = NPZDataset("lafem/data/cir3_e64_64_c8/inclusion", [str(i) for i in range(10000, 10200)])
dataset = StackDataset(gd_set, label_set)
print(dataset[0])
loader = DataLoader(dataset, batch_size=100,
                    shuffle=True, num_workers=0, pin_memory=True)

REPEAT = 1
save_dir = 'lafem/'
use_noise_filter = True

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
        for gd, label in tqdm(loader, desc='Validation', unit='batch', leave=False, position=2):
            gd = gd.clone()
            N = gd.shape[0]
            x = torch.stack([gd, gn[None, ...].repeat(N, 1, 1)], dim=2)
            noise = torch.randn_like(x[:, :, 0, :]) * noise_coef
            if noise_filter:
                noise = noise_filter(noise)
            noise = x[:, :, 0, :] * noise
            x[:, :, 0, :] += noise
            y_pred = model(x).squeeze(1)
            label = label[0].reshape(y_pred.shape).to(dtype=torch.float32)
            y_pred = transform(y_pred) if transform else y_pred
            label = transform(label) if transform else label
            loss += loss_fn(y_pred, label).detach().cpu().item()
            count += 1

    return loss / count


### Validation and Visualization Scripts ###

x = np.linspace(-1, 1, 64)
y = np.linspace(-1, 1, 64)
X, Y = np.meshgrid(x, y, indexing='ij')
tag, type_, noise_coef, noise_filter, ckpts_path = settings[0]

gamma1 = np.linspace(0., 1., 20)
# gamma2 = np.linspace(0., 1., 20)
# G1, G2 = np.meshgrid(gamma1, gamma2, indexing='ij')
result = []

model, MODEL_NAME = build_eit_model(
    name = 'unet100',
    ext = 63,
    n_channel = 8,
    tag = tag,
    fractype = type_,
    eigen_file = "lafem/data/laplace_beltrami_63_63.npz",
    ckpts_path = "lafem/ckpts",
    device = device
)
model.df_solver.bc_filter.gamma.requires_grad_(False)
model.eval()

for g1 in tqdm(gamma1, position=0, ascii=True):
    # for g2 in tqdm(gamma2, position=1, ascii=True):
        model.df_solver.bc_filter.gamma.copy_(torch.tensor(g1, dtype=torch.float64))
        # model.df_solver.bc_filter.gamma[1] = g2

        cross_entropy_loss = validate(
            model, loader, F.binary_cross_entropy_with_logits, noise_coef,
            noise_filter, transform=None, repeat=REPEAT
        )

        result.append(cross_entropy_loss)

result = np.array(result).reshape(20,)
print(result)
np.save(f'lafem/landscape_1_{tag}.npy', result)
# CenterCrop(32)
