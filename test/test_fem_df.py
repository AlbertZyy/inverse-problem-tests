"""Calculate the residual in the data feature solution."""

import sys
sys.path.append("./src")

import numpy as np
import torch
from torch.utils.data import RandomSampler, BatchSampler
from fealpy.torch.mesh import TriangleMesh
from tqdm import tqdm

from fem import DataFeatureFEMSolver, EITDataPreprocessor, LaplaceFEMSolver
from dataset import NPYDataset


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
EXT = 63

mesh = TriangleMesh.from_box([-1, 1, -1, 1], nx=EXT, ny=EXT, device=DEVICE)
solver = LaplaceFEMSolver(mesh, p=1, reserve_matrix=True)
df_prepro = EITDataPreprocessor(solver)
df_solver = DataFeatureFEMSolver(solver)
dataset = NPYDataset('data/cir3_e64_64_c8/gd', names=[str(i) for i in range(200)])
gn_ = torch.from_numpy(np.load('data/cir3_e64_64_c8/gn.npy'))

sampler = RandomSampler(dataset)
batch_sampler = BatchSampler(sampler, batch_size=100, drop_last=False)

vuh_res = []
phi_res = []

for indices in tqdm(batch_sampler):
    gd = dataset.__getitems__(indices)
    gn = gn_.unsqueeze(0).broadcast_to(gd.shape)
    data = torch.stack([gd, gn], dim=-2).to(DEVICE)
    gnvn = df_prepro(data)
    vuh_res.append(solver.residual_fd(df_prepro.vuh).item())
    phi = df_solver(gnvn)
    phi_res.append(solver.residual_fn(df_solver.img).item())

print(sum(vuh_res) / len(vuh_res))
print(sum(phi_res) / len(phi_res))
