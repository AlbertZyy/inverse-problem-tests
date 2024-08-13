"""Calculate the residual in the data feature solution."""

import sys
sys.path.append("./src")

import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from fealpy.torch.mesh import TriangleMesh
from tqdm import trange

from fractional import Fractional
from fem import DataFeatureFEMSolver, EITDataPreprocessor, LaplaceFEMSolver


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
EXT = 63
S0 = -0.222

### Generate a phi as the target solution

frac = Fractional(252, device=DEVICE)
frac.from_npz(r"data/laplace_beltrami_63_63_torch.npz")
frac.initialize(S0)
frac.requires_grad_(False)

mesh = TriangleMesh.from_box([-1, 1, -1, 1], nx=EXT, ny=EXT, device=DEVICE)
solver = LaplaceFEMSolver(mesh, p=1, reserve_matrix=True)
df_prepro = EITDataPreprocessor(solver)
df_solver = DataFeatureFEMSolver(solver, frac)
gd = torch.from_numpy(np.load('data/cir3_e64_64_c8/gd/50.npy'))
gn = torch.from_numpy(np.load('data/cir3_e64_64_c8/gn.npy'))

gn = gn.broadcast_to(gd.shape)
data = torch.stack([gd, gn], dim=-2).to(DEVICE)
gnvn = df_prepro(data[None, ...])
target_phi = df_solver(gnvn).detach()

### Train a new s to see if converged

frac.initialize(0.0)
frac.requires_grad_(True)
criterion = MSELoss()
optim = Adam((frac.s,), lr=0.01, betas=(0.9, 0.98))

s_list = []

for i in trange(0, 200):
    optim.zero_grad()
    phi = df_solver(gnvn)
    loss = (phi - target_phi).pow(2).sum().sqrt()
    loss.backward()
    optim.step()

    s_list.append(frac.s.detach().item())

from matplotlib import pyplot as plt

plt.plot(s_list, color='b')
plt.plot([0, len(s_list)], [S0, S0], '--', color='b')
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('S', fontsize=16)
plt.show()
