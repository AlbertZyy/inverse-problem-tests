"""Calculate the residual in the data feature solution."""

import sys
sys.path.append("./src")

import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from fealpy.torch.mesh import TriangleMesh
from tqdm import trange
import json
import time

from fractional import Fractional
from fem import DataFeatureFEMSolver, EITDataPreprocessor, LaplaceFEMSolver


torch.manual_seed(202400)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
EXT = 63


def main(s0, noise, domain):
    ### Generate a phi as the target solution

    frac = Fractional(252, device=DEVICE)
    frac.from_npz(r"data/laplace_beltrami_63_63_torch.npz")
    frac.initialize(s0)
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

    # get the range of ipoints

    if domain == 'sub':
        ips = mesh.interpolation_points(p=1)
        flag1 = torch.logical_and(ips[:, 0] > -0.1, ips[:, 0] < 0.1)
        flag2 = torch.logical_and(ips[:, 1] > -0.1, ips[:, 1] < 0.1)
        flag = torch.logical_and(flag1, flag2).nonzero(as_tuple=True)[0]
    elif domain == 'full':
        flag = slice(None)
    else:
        raise ValueError(f"DOMAIN must be 'sub' or 'full', but got {domain}")

    print(flag)

    frac.initialize(0.0)
    frac.requires_grad_(True)
    criterion = MSELoss()
    optim = Adam((frac.s,), lr=0.01, betas=(0.9, 0.98))

    s_list = []

    for i in trange(0, 300):
        optim.zero_grad()
        noisy_gnvn = gnvn + torch.randn_like(gnvn) * gnvn * noise
        phi = df_solver(noisy_gnvn)
        loss = criterion(phi[..., flag], target_phi[..., flag])
        loss.backward()
        optim.step()

        s_list.append(frac.s.detach().item())


    data = {
        's0': s0,
        'noise': noise,
        'domain': domain,
        'loss': s_list,
    }

    with open(f'test_fem_autograd/plot_data/loss_{time.time()}.json', 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":

    for s0 in [0.005, 0.495, 0.995]:
        for noise in [0.0, 0.05, 0.1, 0.2]:
            main(s0, noise, "sub")
