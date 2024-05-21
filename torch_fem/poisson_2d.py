
import torch
from torch import Tensor

CONTEXT = 'torch'

if CONTEXT == 'torch':
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

elif CONTEXT == 'numpy':
    from fealpy.mesh import TriangleMesh
    from fealpy.functionspace import LagrangeFESpace
    from fealpy.fem import (
        BilinearForm, LinearForm,
        ScalarDiffusionIntegrator,
        ScalarSourceIntegrator,
        DirichletBC
    )
    from scipy.sparse.linalg import cg

    from numpy import cos, pi

else:
    raise ValueError('Unknow context: %s' % CONTEXT)

from fealpy.ml import timer
from matplotlib import pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NX, NY = 128, 128

def source(points: Tensor):
    x = points[..., 0]
    y = points[..., 1]
    return 2*pi**2 * cos(pi*x) * cos(pi*y)


def solution(points: Tensor):
    x = points[..., 0]
    y = points[..., 1]
    return cos(pi*x) * cos(pi*y)

tmr = timer()

if CONTEXT == 'torch':
    mesh_numpy = TMD.from_box(nx=NX, ny=NY)
    cell = mesh_numpy.ds.cell
    node = mesh_numpy.node
    next(tmr)
    mesh = TriangleMesh(
        torch.from_numpy(node).to(device),
        torch.from_numpy(cell).to(device),
    )
elif CONTEXT == 'numpy':
    next(tmr)
    mesh = TriangleMesh.from_box(nx=NX, ny=NY)
    mesh_numpy = mesh


space = LagrangeFESpace(mesh, p=3)

tmr.send('mesh_and_space')

bform = BilinearForm(space)
bform.add_domain_integrator(ScalarDiffusionIntegrator())

lform = LinearForm(space)
lform.add_domain_integrator(ScalarSourceIntegrator(source))

tmr.send('forms')

torch.cuda.default_stream().synchronize()


with torch.cuda.nvtx.range("Assembly A"):
    A = bform.assembly()

with torch.cuda.nvtx.range("Assembly F"):
    F = lform.assembly()

if CONTEXT == 'torch':
    F = F.to_dense()
    uh = torch.zeros((space.number_of_global_dofs(), ), dtype=torch.float64, device=device)
elif CONTEXT == 'numpy':
    uh = space.function()

tmr.send('assembly')
with torch.cuda.nvtx.range("Apply dirichlet BC"):
    A, F = DirichletBC(space, solution).apply(A, F, uh)

tmr.send('dirichlet')

if CONTEXT == 'torch':
    A = A.to_sparse_csr()
    with torch.cuda.nvtx.range("Solve CG"):
        uh = sparse_cg(A, F, uh, maxiter=5000)
    uh = uh.detach().cpu().numpy()
elif CONTEXT == 'numpy':
    uh, info = cg(A, F, uh)

tmr.send('solve(cg)')
tmr.send('stop')

# fig = plt.figure()
# axes = fig.add_subplot(111, projection='3d')
# mesh_numpy.show_function(axes, uh, cmap='jet')
# plt.show()
