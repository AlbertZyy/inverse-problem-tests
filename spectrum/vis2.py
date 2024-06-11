
import sys

sys.path.append('./src')

import numpy as np
import torch
from torch import Tensor
from matplotlib import pyplot as plt

from fealpy.torch.mesh import TriangleMesh

from fractional import Fractional
from fem import LaplaceFEMSolver, EITDataPreprocessor
from dataset import NPYDataset


def get_energy(alpha: Tensor):
    return alpha.reshape(8, 126, 2).pow(2).sum(-1) # (8, 126)


NOISE = 0.01
L_FACTOR = 8.4
ENABLE_FILTER = True
EXT = 63
H = 2./EXT

frac = Fractional(252)
frac.from_npz('data/laplace_beltrami_63_63.npz')
frac.initialize(0.75)
freq = frac.w[::2].sqrt()

mesh = TriangleMesh.from_box([-1, 1, -1, 1], EXT, EXT)
solver = LaplaceFEMSolver(mesh, p=1)
df_prepro = EITDataPreprocessor(solver)

dataset = NPYDataset('data/cir3_e64_64_c8/gd', names=[str(i) for i in range(100)])
gd = dataset[15]
gn = torch.from_numpy(np.load('data/cir3_e64_64_c8/gn.npy'))
data = torch.stack([gd, gn], dim=-2)

gnvn = df_prepro(data[None, ...]).squeeze_(0)

energy1 = get_energy(frac.decompose(gnvn))

gnvn2 = frac(gnvn)
energy2 = get_energy(frac.decompose(gnvn2))

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

plt.savefig('spectrum/figures/vis2_torch.png')

plt.show()
