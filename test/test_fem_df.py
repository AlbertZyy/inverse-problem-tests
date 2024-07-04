"""Calculate the residual in the data feature solution."""

import sys
sys.path.append("./src")

import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from fealpy.torch.mesh import TriangleMesh

from fractional import Fractional
from fem import DataFeatureFEMSolver, EITDataPreprocessor, LaplaceFEMSolver


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
EXT = 63

frac = Fractional(252, device=DEVICE)
frac.from_npz(r"data/laplace_beltrami_63_63_torch.npz")
frac.initialize(-0.87)
frac.requires_grad_(False)

mesh = TriangleMesh.from_box([-1, 1, -1, 1], nx=EXT, ny=EXT, device=DEVICE)
solver = LaplaceFEMSolver(mesh, p=1, reserve_matrix=True)
df_prepro = EITDataPreprocessor(solver)
df_solver = DataFeatureFEMSolver(solver, frac)
# dataset = NPYDataset('data/cir3_e64_64_c8/gd', names=[str(i) for i in range(200)])
gd = torch.from_numpy(np.load('data/cir3_e64_64_c8/gd/50.npy'))
gn = torch.from_numpy(np.load('data/cir3_e64_64_c8/gn.npy'))

gn = gn.broadcast_to(gd.shape)
data = torch.stack([gd, gn], dim=-2).to(DEVICE)
gnvn = df_prepro(data[None, ...])
target_phi = df_solver(gnvn).detach()

frac.initialize(0.0)
frac.requires_grad_(True)
criterion = MSELoss()
optim = Adam((frac.s,), lr=1e-2)


for i in range(0, 200):
    optim.zero_grad()
    phi = df_solver(gnvn)
    loss = (phi - target_phi).pow(2).sum().sqrt()
    loss.backward()
    optim.step()

    if i % 10 == 0:
        print(f"{i}: {loss.item()}")
        print(frac.s.detach().item())
