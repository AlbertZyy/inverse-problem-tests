
import sys

sys.path.append('./src')

import numpy as np
import torch
from torch import Tensor
from torch.nn import BatchNorm1d
from matplotlib import pyplot as plt

from fealpy.mesh import TriangleMesh
from lafemeit.model import DataPreprocessor, Fractional
from lafemeit.solver import LaplaceFEMSolver

from fractional import Fractional
from dataset import NPYDataset, NPZDataset


def get_energy(alpha: Tensor):
    return alpha.reshape(8, 126, 2).pow(2).sum(-1) # (8, 126)


NOISE = 0.01
L_FACTOR = 8.4
ENABLE_FILTER = True
EXT = 63
H = 2./EXT

frac = Fractional(252)
frac.from_npz('lafem/data/laplace_beltrami_63_63.npz')
frac.initialize(0.75)
freq = frac.w[::2].sqrt()

mesh = TriangleMesh.from_box([-1, 1, -1, 1], EXT, EXT)
solver = LaplaceFEMSolver(mesh, p=1)
df_prepro = DataPreprocessor(solver)

###
# cdata = NPZDataset('data/gdgn_cir3_e64_64_c8_train', names=[str(i) for i in range(100)],
#                    channel_keys=['1', '2', '3', '4', '5', '6', '8', '16'])
# gn0 = cdata[0][0][:, 1, :]
###

# bn = BatchNorm1d(8, affine=False, dtype=mesh.ftype)

dataset = NPYDataset('lafem/data/cir3_e64_64_c8/gd', names=[str(i) for i in range(100)])
gd = dataset[26]
gn = torch.from_numpy(np.load('lafem/data/cir3_e64_64_c8/gn.npy'))
data = torch.stack([gd, gn], dim=-2)

gnvn = df_prepro(data[None, ...]).squeeze(0)

# gnvn = bn(gnvn)

energy1 = get_energy(frac.decompose(gnvn))

gnvn2 = frac(gnvn)
energy2 = get_energy(frac.decompose(gnvn2))

fig = plt.figure(figsize=(15, 5))
fig.tight_layout()

axes = fig.add_subplot(1, 2, 1)
axes.plot(freq.detach(), energy1.detach().numpy().T, linewidth=0.75)
axes.legend(['$l = 1$', '$l = 2$', '$l = 3$', '$l = 4$', '$l = 5$', '$l = 6$', '$l = 8$', '$l = 16$'])
axes.set_xscale('log')
axes.set_yscale('log')
axes.set_ylim(1e-14, 1e-1)
axes.grid(True)
axes.set_xlabel('Frequency', fontsize=14)
axes.set_ylabel('Energy', fontsize=14)
axes.set_title(r'$\xi_l$', fontsize=16)

axes = fig.add_subplot(1, 2, 2)
axes.plot(freq.detach(), energy2.detach().numpy().T, linewidth=0.75)
axes.legend(['$l = 1$', '$l = 2$', '$l = 3$', '$l = 4$', '$l = 5$', '$l = 6$', '$l = 8$', '$l = 16$'])
axes.set_xscale('log')
axes.set_yscale('log')
axes.set_ylim(1e-14, 1e-1)
axes.grid(True)
axes.set_xlabel('Frequency', fontsize=14)
axes.set_ylabel('Energy', fontsize=14)
axes.set_title(r'$(-\Delta_{\partial\Omega})^{0.75}(\xi_l)$', fontsize=16)

plt.savefig('spectrum/figures/vis2.png')

plt.show()
