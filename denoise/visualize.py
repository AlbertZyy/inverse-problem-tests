
import sys
from typing import Callable

sys.path.append('./src')

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import MSELoss, Module
from tqdm import tqdm

from cnn import build_model
from dataset import NPZDataset
from common import add_gaussian_noise

NOISE = 0.01
NO_EVALUATE = True

model_1, name_1 = build_model('cpu', '')
model_1.eval()


if not NO_EVALUATE:
    validation_set = NPZDataset('./data/gdgn_64_64_validate/', 200)
    loader = DataLoader(validation_set, batch_size=100, shuffle=False)
    mse = MSELoss()

    def validate(model: Module, loader: DataLoader,
                loss_fn: Callable[[Tensor, Tensor], Tensor]):
        model.eval()
        loss = 0
        count = 0

        for x, _ in tqdm(loader, desc='Validation', unit='batch'):
            BATCH, CHANNEL, _, BDDOF = x.shape
            x_o = x.clone()
            g = torch.randn((BATCH, ))
            noise = torch.exp(g) * NOISE
            add_gaussian_noise(x[:, :, 0, :], noise)
            y_pred = model(x.reshape(BATCH, CHANNEL*2, BDDOF))
            loss += loss_fn(y_pred, x_o.reshape(BATCH, CHANNEL*2, BDDOF)).item()
            count += 1

        return loss / count

    for model, name in zip([model_1, ], [name_1, ]):
        mse_loss = validate(model, loader, mse)
        print(f'Validation loss for {name}')
        print(f'  - mse loss: {mse_loss}')


from fractional import Fractional
from data_feature import MultiChannelDataFeature


EXT = 63
H = 2./EXT

frac = Fractional(252)
frac.from_npz(r"./data/laplace_beltrami_63_63.npz")
df = MultiChannelDataFeature.from_domain([EXT, EXT], [H, H], frac)

frac.initialize(s=0.57)

dataset = NPZDataset(r"./data/gdgn_64_64_train", 100)
gdgn = dataset[81][0]

CHANNEL, _, BDDOF = gdgn.shape

alpha_1 = frac.alpha(gdgn[:, 0, :])
add_gaussian_noise(gdgn[:, 0, :], NOISE)
alpha_2 = frac.alpha(gdgn[:, 0, :])
denoised = model_1(gdgn.reshape(1, CHANNEL*2, BDDOF)).reshape(CHANNEL, 2, BDDOF)
alpha_3 = frac.alpha(denoised[:, 0, :])


def sum_energy(x: Tensor) -> Tensor:
    CHANNEL = x.shape[0]
    x = x.reshape(CHANNEL, -1, 2)
    return x.norm(dim=-1)


from matplotlib import pyplot as plt
PI = torch.pi

length = alpha_1.shape[1] // 2
x = torch.arange(1, length+1, dtype=torch.float32) * PI/4

fig = plt.figure('data_feature_bd_spectrum', figsize=[18, 6])
fig.suptitle("Energy spectrum of boundary value "
             f"(s={frac.s.item():.2f})")

axes = fig.add_subplot(1, 3, 1)
axes.plot(x, sum_energy(alpha_2).T.detach())
axes.grid(True)
axes.set_xlabel('Frequency')
axes.set_xscale('log')
axes.set_ylim(1e-6, 1e+2)
axes.set_ylabel('Energy')
axes.set_yscale('log')
axes.set_title('Spectrum of the noisy gd')
axes.legend([1, 2, 3, 4, 5, 6, 8, 16])

axes = fig.add_subplot(1, 3, 2)
axes.plot(x, sum_energy(alpha_3).T.detach())
axes.grid(True)
axes.set_xlabel('Frequency')
axes.set_xscale('log')
axes.set_ylim(1e-6, 1e+2)
axes.set_ylabel('Energy')
axes.set_yscale('log')
axes.set_title('Spectrum of denoised gd')

axes = fig.add_subplot(1, 3, 3)
axes.plot(x, sum_energy(alpha_1).T.detach())
axes.grid(True)
axes.set_xlabel('Frequency')
axes.set_xscale('log')
axes.set_ylim(1e-6, 1e+2)
axes.set_ylabel('Energy')
axes.set_yscale('log')
axes.set_title('Spectrum of the real gd')

plt.show()
