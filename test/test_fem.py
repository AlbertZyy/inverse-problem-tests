
import sys
sys.path.append("./src")

import torch
from torch import Tensor, sin, tensordot
from fealpy.torch.mesh import TriangleMesh

from fem import LaplaceFEMSolver


BATCH_SIZE = 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

mesh = TriangleMesh.from_box([-1, 1, -1, 1], nx=32, ny=32, device=device)
solver = LaplaceFEMSolver(mesh, p=3)


def neumann(p: Tensor):
    x = p[..., 0]
    y = p[..., 1]
    theta = torch.atan2(y, x)
    omega = torch.arange(1, BATCH_SIZE+1, dtype=p.dtype, device=p.device)
    return sin(tensordot(omega, theta, dims=0))

uh = solver.solve_from_gn(neumann, batch_size=BATCH_SIZE)

# pts = solver.space.interpolation_points()[solver.bd_dof_flag, :]
original = solver._latest_gn_f[:, solver.bd_dof_flag] # (B, bddof)
calculated = solver.normal_derivative(uh)

err_bddof = solver.solve_from_gnf(calculated, -original, f_only=True)
err_bddof = err_bddof[:, solver.bd_dof_flag]
print(err_bddof.pow(2).mean(dim=-1).sqrt().mean(dim=0).item())

uh2 = solver.solve_from_gnf(calculated)
err = (uh2 - uh).pow(2).mean(dim=-1).sqrt()
print(err.mean().item())
