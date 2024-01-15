
import sys
sys.path.append("./src")

import torch
from torch import Tensor

from fractional import FractionalWithHighcut
from data_feature import MultiChannelDataFeature
from dataset import NPZDataset
from common import add_gaussian_noise

NOISE = 0.0
EXT = 63
H = 2./EXT

frac = FractionalWithHighcut(252)
frac.from_npz(r"./data/laplace_beltrami_63_63.npz")
df = MultiChannelDataFeature.from_domain([EXT, EXT], [H, H], frac)

# 20% - 391.72
# 50% - 2448.27
frac.initialize(s=0.57, hc=1e+5)

dataset = NPZDataset(r"./data/gdgn_64_64_train", 100)
gdgn = dataset[81][0]

add_gaussian_noise(gdgn[:, 0, :], NOISE)

gnvn = df.gd2gn_diff(gdgn[None, ...])

alpha_0 = frac.alpha(gdgn[:, 0, :])
alpha_1 = frac.alpha(gnvn)[0, ...]
alpha_2 = frac.alpha(frac(gnvn))[0, ...]


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
             f"(s={frac.s.item():.2f}, noise={NOISE:.2e})")

axes = fig.add_subplot(1, 3, 1)
axes.plot(x, sum_energy(alpha_0).T.detach())
axes.grid(True)
axes.set_xlabel('Frequency')
axes.set_xscale('log')
axes.set_ylim(1e-6, 1e+2)
axes.set_ylabel('Energy')
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
axes.set_title('Spectrum of gn-vn')

axes = fig.add_subplot(1, 3, 3)
axes.plot(x, sum_energy(alpha_2).T.detach())
axes.grid(True)
axes.set_xlabel('Frequency')
axes.set_xscale('log')
axes.set_ylim(1e-6, 1e+2)
axes.set_ylabel('Energy')
axes.set_yscale('log')
axes.set_title('Spectrum of after-frac gn-vn')

plt.show()
