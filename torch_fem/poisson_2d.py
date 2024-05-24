
import torch
from torch import Tensor

CONTEXT = 'torch'

from fealpy.mesh import TriangleMesh as TMD
from fealpy.torch.mesh import TriangleMesh
from fealpy.torch.functionspace import LagrangeFESpace
from fealpy.torch.fem import (
    BilinearForm, LinearForm,
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    DirichletBC
)
from fealpy.torch.solver import sparse_cg

from torch import cos, pi

from fealpy.ml import timer
from matplotlib import pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NX, NY = 64, 64

def source(points: Tensor):
    x = points[..., 0]
    y = points[..., 1]
    kwargs = {'dtype': points.dtype, "device": points.device}
    coef = torch.linspace(pi/2, 5*pi, 10).to(**kwargs)
    return torch.einsum(
        "b, ...b -> ...b",
        2*coef**2,
        cos(torch.tensordot(x, coef, dims=0)) * cos(torch.tensordot(y, coef, dims=0))
    )


def solution(points: Tensor):
    x = points[..., 0]
    y = points[..., 1]
    kwargs = {'dtype': points.dtype, "device": points.device}
    coef = torch.linspace(pi/2, 5*pi, 10).to(**kwargs)
    return cos(torch.tensordot(x, coef, dims=0)) * cos(torch.tensordot(y, coef, dims=0))


tmr = timer()

mesh_numpy = TMD.from_box(nx=NX, ny=NY)
cell = mesh_numpy.ds.cell
node = mesh_numpy.node
next(tmr)
mesh = TriangleMesh(
    torch.from_numpy(node).to(device),
    torch.from_numpy(cell).to(device),
)


space = LagrangeFESpace(mesh, p=3)

tmr.send('mesh_and_space')

bform = BilinearForm(space)
bform.add_domain_integrator(ScalarDiffusionIntegrator())

lform = LinearForm(space, batch_size=10)
lform.add_domain_integrator(ScalarSourceIntegrator(source, batched=True))

tmr.send('forms')


with torch.cuda.nvtx.range("Assembly A"):
    A = bform.assembly()

with torch.cuda.nvtx.range("Assembly F"):
    F = lform.assembly()

uh = torch.zeros((space.number_of_global_dofs(), 10), dtype=torch.float64, device=device)

tmr.send('assembly')
with torch.cuda.nvtx.range("Apply dirichlet BC"):
    A, F = DirichletBC(space, solution).apply(A, F, uh)

tmr.send('dirichlet')

A = A.to_sparse_csr()
with torch.cuda.nvtx.range("Solve CG"):
    uh = sparse_cg(A, F, uh, maxiter=5000)
uh = uh.detach().cpu().numpy()


tmr.send('solve(cg)')
tmr.send('stop')

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
mesh_numpy.show_function(axes, uh[:, 3], cmap='jet')
plt.show()
