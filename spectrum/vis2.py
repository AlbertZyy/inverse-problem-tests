
import sys

sys.path.append('./src')

import torch
from torch import Tensor
from matplotlib import pyplot as plt

from fractional import Fractional
from data_feature import MultiChannelDataFeature
from fdm import LaplaceFDMSolver
from dataset import TPZDataset, NPZDataset


def get_energy(alpha: Tensor):
    return alpha.reshape(8, 126, 2).pow(2).sum(-1) # (8, 126)


NOISE = 0.01
L_FACTOR = 8.4
ENABLE_FILTER = True
EXT = 63
H = 2./EXT

frac = Fractional(252)
frac.from_npz(r'data\laplace_beltrami_63_63.npz')
frac.initialize(-0.75)
freq = frac.w[::2].sqrt()

df_solver = MultiChannelDataFeature(
    LaplaceFDMSolver([EXT, EXT], [H, H]),
    Fractional(252)
)
df_solver._frac.from_npz(r'data\laplace_beltrami_63_63.npz')
df_solver._frac.initialize(0.75)
dataset = NPZDataset(r'data\gdgn_cir3_e64_64_c8_validate', names=10,
                     channel_keys=['1', '2', '3', '4', '5', '6', '8', '16'])
data = dataset[0][0]
gnvn = df_solver.gd2gn_diff(data[None, ...])[0]
energy1 = get_energy(frac.decompose(gnvn)).sqrt_()
gnvn2 = df_solver._frac(gnvn)
energy2 = get_energy(frac.decompose(gnvn2)).sqrt_()

fig = plt.figure(figsize=(15, 5))
fig.tight_layout()

axes = fig.add_subplot(1, 2, 1)
axes.plot(freq.detach(), energy1.detach().numpy().T, linewidth=1)
axes.legend(['1', '2', '3', '4', '5', '6', '8', '16'])
axes.set_xscale('log')
axes.set_yscale('log')
axes.grid(True)
axes.set_xlabel('Frequency', fontsize=14)
axes.set_ylabel('Energy', fontsize=14)
axes.set_title('Spectrum of gn-vn', fontsize=16)

axes = fig.add_subplot(1, 2, 2)
axes.plot(freq.detach(), energy2.detach().numpy().T, linewidth=1)
axes.legend(['1', '2', '3', '4', '5', '6', '8', '16'])
axes.set_xscale('log')
axes.set_yscale('log')
axes.grid(True)
axes.set_xlabel('Frequency', fontsize=14)
axes.set_ylabel('Energy', fontsize=14)
axes.set_title('Spectrum of gn-vn after frac-LB (s=0.75)', fontsize=16)

plt.savefig('spectrum/figures/vis2.png')

plt.show()
