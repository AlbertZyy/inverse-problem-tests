
import sys
sys.path.append("./src")

import torch
from torch import Tensor

from fractional import MultiChannelFractional, FractionalWithHighcut
from dataset import NPZDataset
from common import add_gaussian_noise

USE_NOISE = False
NOISE = 0.001

frac = FractionalWithHighcut(252)
# frac = MultiChannelFractional(252, 8, 3.2)
frac.from_npz(r"./data/laplace_beltrami_63_63.npz")
# frac.initialize(s=0.5, hc=391.72)

# 20% - 391.72
# 50% - 2448.27
frac.initialize(s=1.0, hc=1e+5)

dataset = NPZDataset(r"./data/gdgn_64_64_train", 100)
gd = dataset[81][0][:, 0, :]

if USE_NOISE:
    gd = add_gaussian_noise(gd, NOISE, noise_only=True)
else:
    add_gaussian_noise(gd, NOISE)

alpha = frac.alpha(gd)
val = frac(gd)
alpha_2 = frac.alpha(val)


def sum_energy(x: Tensor) -> Tensor:
    CHANNEL = x.shape[0]
    x = x.reshape(CHANNEL, -1, 2)
    return x.norm(dim=-1)


from matplotlib import pyplot as plt
PI = torch.pi

length = alpha.shape[1] // 2
x = torch.arange(1, length+1, dtype=torch.float32) * PI/4

fig = plt.figure()

axes = fig.add_subplot(1, 2, 1)
axes.plot(x, sum_energy(alpha).T.detach())
axes.grid(True)
axes.set_xlabel('Frequency')
axes.set_xscale('log')
axes.set_ylim(1e-6, 1e+2)
axes.set_ylabel('Energy')
axes.set_yscale('log')
axes.set_title('Original Spectrum')
axes.legend([1, 2, 3, 4, 5, 6, 8, 16])

axes = fig.add_subplot(1, 2, 2)
axes.plot(x, sum_energy(alpha_2).T.detach())
axes.grid(True)
axes.set_xlabel('Frequency')
axes.set_xscale('log')
axes.set_ylim(1e-6, 1e+2)
axes.set_ylabel('Energy')
axes.set_yscale('log')
axes.set_title('Reconstructed Spectrum')

plt.show()
