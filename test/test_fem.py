
import sys
sys.path.append("./src")

import torch
from torch import Tensor, sin, tensordot
from fealpy.torch.mesh import TriangleMesh

from fem import LaplaceFEMSolver


BATCH_SIZE = 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

mesh = TriangleMesh.from_box([-1, 1, -1, 1], nx=64, ny=64, device=device)
solver = LaplaceFEMSolver(mesh)


def neumann(p: Tensor):
    x = p[..., 0]
    y = p[..., 1]
    theta = torch.atan2(y, x)
    omega = torch.arange(1, BATCH_SIZE+1, dtype=p.dtype, device=p.device)
    return sin(tensordot(omega, theta, dims=0))

uh = solver.solve_from_gn(neumann, batch_size=BATCH_SIZE)

source_val = solver.bsi._source_val
calculated = solver.normal_derivative(uh)

errqf = (calculated - source_val)**2

from fealpy.torch.functional import integral

bcs, ws, phi, fm, index = solver.bsi.fetch(solver.space)
errf = integral(errqf, ws, fm, entity_type=True)
print(errf.sum(1).sqrt())
