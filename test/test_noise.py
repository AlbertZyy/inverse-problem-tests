
import sys
sys.path.append("./src")

import torch
from torch import Tensor

from fractional import Fractional
from dataset import NPZDataset

NOISE = 1.51
EXT = 63
H = 2./EXT

frac = Fractional(252)
frac.from_npz(r"./data/laplace_beltrami_63_63.npz")
frac.initialize(s=0.57)

dataset = NPZDataset(r"./data/gdgn_64_64_train", 100)
gdgn = dataset[72][0]

noise_filter = Fractional(252)
noise_filter.from_npz(r"./data/laplace_beltrami_63_63.npz")
noise_filter.initialize(s=-0.75)
noise_filter.s.requires_grad_(False)

noise = torch.randn_like(gdgn[:, 0, :]) * NOISE
noise = noise_filter(noise)
noise = gdgn[:, 0, :] * noise

alpha_0 = frac.alpha(gdgn[:, 0, :])
alpha_1 = frac.alpha(noise)

gdgn[:, 0, :] += noise
alpha_2 = frac.alpha(gdgn[:, 0, :])


def sum_energy(x: Tensor) -> Tensor:
    CHANNEL = x.shape[0]
    x = x.reshape(CHANNEL, -1, 2)
    return x.norm(dim=-1)

def total_energy(x: Tensor) -> Tensor:
    CHANNEL = x.shape[0]
    x = x.reshape(CHANNEL, -1)
    return x.norm(dim=-1)

energy_data = total_energy(alpha_0)
energy_noise = total_energy(alpha_1)
energy_total = total_energy(alpha_2)

print(f"Energy of data: {energy_data.mean().item():.4e}")
print(f"Energy of noise: {energy_noise.mean().item():.4e}")
print(f"Energy of total: {energy_total.mean().item():.4e}")

print(f"NOISE/DATA: {100*energy_noise.mean().item() / energy_data.mean().item():.2f}%")


from matplotlib import pyplot as plt
PI = torch.pi

length = alpha_1.shape[1] // 2
x = torch.arange(1, length+1, dtype=torch.float32) * PI/4

fig = plt.figure('data_feature_bd_spectrum', figsize=[18, 6])
fig.suptitle("Spectrum of boundary value "
             f"(noise={NOISE:.2e})")

axes = fig.add_subplot(1, 3, 1)
axes.plot(x, sum_energy(alpha_0).T.detach())
axes.grid(True)
axes.set_xlabel('Frequency')
axes.set_xscale('log')
axes.set_ylim(1e-6, 1e+2)
axes.set_ylabel('Amplitude')
axes.set_yscale('log')
axes.set_title('Spectrum of gd')
axes.legend([1, 2, 3, 4, 5, 6, 8, 16])

axes = fig.add_subplot(1, 3, 2)
axes.plot(x, sum_energy(alpha_1).T.detach())
axes.grid(True)
axes.set_xlabel('Frequency')
axes.set_xscale('log')
axes.set_ylim(1e-6, 1e+2)
axes.set_ylabel('Energy')
axes.set_yscale('log')
axes.set_title('Spectrum of noise')

axes = fig.add_subplot(1, 3, 3)
axes.plot(x, sum_energy(alpha_2).T.detach())
axes.grid(True)
axes.set_xlabel('Frequency')
axes.set_xscale('log')
axes.set_ylim(1e-6, 1e+2)
axes.set_ylabel('Energy')
axes.set_yscale('log')
axes.set_title('Spectrum of noisy gd')

plt.show()
