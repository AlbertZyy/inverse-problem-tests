
import sys

sys.path.append('./src')

import torch
from torch import Tensor
from matplotlib import pyplot as plt

from fractional import Fractional
from dataset import TPZDataset, NPZDataset


def get_energy(alpha: Tensor):
    return alpha.reshape(8, 126, 2).pow(2).sum(-1).mean(0) # (126, )


NOISE = 0.01
L_FACTOR = 8.4
ENABLE_FILTER = True

frac = Fractional(252)
frac.from_npz(r'data\laplace_beltrami_63_63.npz')
frac.initialize(-0.75)

dataset = NPZDataset(r'data\gdgn_cir3_e64_64_c8_validate', names=10,
                     channel_keys=['1', '2', '3', '4', '5', '6', '8', '16'])
data = dataset[0][0]
torch.random.manual_seed(2024)
G = torch.randn_like(data[:, 0, :]) * NOISE
gnoise = data[:, 0, :] * G
lnoise = data[:, 0, :] * frac(G) * L_FACTOR
genergy = get_energy(frac.decompose(gnoise)).sqrt_()
lenergy = get_energy(frac.decompose(lnoise)).sqrt_()
freq = frac.w[::2].sqrt()

fig = plt.figure(figsize=(10, 5))
axes = fig.add_subplot(1, 1, 1)
axes.plot(freq.detach(), genergy.detach(), color='r', linewidth=1)
axes.plot(freq.detach(), lenergy.detach(), color='b', linewidth=1)
axes.legend(['Gaussian', 'Low-freq'])
axes.set_xscale('log')
axes.set_yscale('log')
axes.grid(True)
axes.set_xlabel('Frequency', fontsize=14)
axes.set_ylabel('Energy', fontsize=14)
axes.set_title('Energy spectrum of noise', fontsize=16)

plt.savefig('spectrum/figures/vis.png')

plt.show()
