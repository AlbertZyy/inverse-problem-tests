
import os
from typing import Callable, Optional
from functools import partial

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.utils.data import StackDataset, DataLoader
from torchvision.transforms import CenterCrop
from tqdm import tqdm
from fealpy.backend import bm
bm.set_backend("pytorch")
from lafemeit.model import build_eit_model, Fractional
from lafemeit.utils import NPZDataset, NPYDataset


def crop_by_ratio(tensor: Tensor, ratio_lower: float, ratio_upper: float) -> Tensor:
    H, W = tensor.shape[-2:]
    x = torch.arange(W, device=tensor.device)
    y = torch.arange(H, device=tensor.device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    mask = torch.abs(X - W / 2) / (W / 2) >= ratio_lower
    mask |= torch.abs(Y - H / 2) / (H / 2) >= ratio_lower
    mask &= torch.abs(X - W / 2) / (W / 2) <= ratio_upper
    mask &= torch.abs(Y - H / 2) / (H / 2) <= ratio_upper
    indices = torch.nonzero(mask, as_tuple=True)
    return tensor[..., indices[0], indices[1]]

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
low_pass = Fractional(252, device=device)
low_pass.from_npz(r"./lafem/data/laplace_beltrami_63_63.npz")
low_pass.initialize(gamma=-0.75)
low_pass.gamma.requires_grad_(False)


settings = [
#   ('tag',       'type', 'noise', 'filter', 'ckpts_path')
    ('nn_nograd', 'nograd', 0.0,  None, 'old/test_no_noise/ckpts'),
    ('nn_single', 'single', 0.0,  None, 'old/test_no_noise/ckpts'),
    ('nn_multi',  'multi',  0.0,  None, 'old/test_no_noise/ckpts'),
]

gd_set = NPYDataset("lafem/data/cir3_e64_64_c8/gd", [str(i) for i in range(10000, 12000)])
gn = torch.from_numpy(np.load('lafem/data/cir3_e64_64_c8/gn.npy')).to(device)
label_set = NPZDataset("lafem/data/cir3_e64_64_c8/inclusion", [str(i) for i in range(10000, 12000)])
dataset = StackDataset(gd_set, label_set)
print(dataset[0])
loader = DataLoader(dataset, batch_size=100,
                    shuffle=True, num_workers=0, pin_memory=True)

REPEAT = 3
save_dir = 'old/test_no_noise/'
use_noise_filter = True

def validate(model: Module,
             loader,
             loss_fn: Callable[[Tensor, Tensor], Tensor],
             noise_coef: float,
             noise_filter: Optional[Module]=None,
             transforms: list[Callable[[Tensor], Tensor]] = [],
             repeat: int=1):
    model.eval()
    count = 0
    loss_list = [0.0] * len(transforms)

    for _ in range(repeat):
        for gd, label in tqdm(loader, desc='Validation', unit='batch'):
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
            for idx, transform in enumerate(transforms):
                loss_list[idx] += loss_fn(
                    transform(y_pred), transform(label)
                ).detach().cpu().item()
            count += 1

    return [loss / count for loss in loss_list]


### Validation and Visualization Scripts ###

model_cursor = 0

result_string = ""
result_rounded = ""
x = np.linspace(-1, 1, 64)
y = np.linspace(-1, 1, 64)
X, Y = np.meshgrid(x, y, indexing='ij')

for tag, type_, noise_coef, noise_filter, ckpts_path in settings:
    model, MODEL_NAME = build_eit_model(
        name = 'unet100',
        ext = 63,
        n_channel = 8,
        tag = tag,
        fractype = type_,
        eigen_file = "lafem/data/laplace_beltrami_63_63.npz",
        ckpts_path = ckpts_path,
        device = device
    )
    model.eval()
    model_cursor += 1 # starts from 1

    cross_entropy_loss = validate(
        model, loader,
        F.binary_cross_entropy_with_logits,
        noise_coef,
        noise_filter,
        transforms=[partial(crop_by_ratio, ratio_lower=ratio_l, ratio_upper=ratio_u)
                    for ratio_l, ratio_u in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]],
        repeat=REPEAT
    )
    print(f'Validation loss for {MODEL_NAME}: {cross_entropy_loss}')
    result_string += f"BCE: {cross_entropy_loss}\n"
    result_rounded += f"Rounded: {[round(data, 5) for data in cross_entropy_loss]}\n"

with open(os.path.join(save_dir, 'show_depth_result.txt'), 'w') as f:
    f.write(result_string)
    f.write('\n')
    f.write(result_rounded)
# CenterCrop(32)