
import numpy as np
import torch
from fealpy.mesh import TriangleMesh as TMD


CONTEXT = 'numpy'

if CONTEXT == 'torch':
    from fealpy.torch.mesh import TriangleMesh
    from fealpy.torch.functionspace import LagrangeFESpace
    from fealpy.torch.fem import (
        BilinearForm, LinearForm,
        ScalarDiffusionIntegrator,
        ScalarSourceIntegrator,
        DirichletBC
    )
    from fealpy.torch.solver import sparse_cg as cg
    from torch import cos, pi

elif CONTEXT == 'numpy':
    from fealpy.np.mesh import TriangleMesh
    from fealpy.np.functionspace import LagrangeFESpace
    from fealpy.np.fem import (
        BilinearForm, LinearForm,
        ScalarDiffusionIntegrator,
        ScalarSourceIntegrator,
        DirichletBC
    )
    from scipy.sparse.linalg import cg
    from numpy import cos, pi

from fealpy.utils import timer
from matplotlib import pyplot as plt

NX, NY = 128, 128

def source(points):
    x = points[..., 0]
    y = points[..., 1]
    return 2*pi**2 * cos(x * pi) * cos(y * pi)


def solution(points):
    x = points[..., 0]
    y = points[..., 1]
    return cos(x * pi) * cos(y * pi)

tmr = timer()

mesh_numpy = TMD.from_box(nx=NX, ny=NY)
next(tmr)

# build mesh and space
mesh = TriangleMesh.from_box(nx=NX, ny=NY)
space = LagrangeFESpace(mesh, p=1)
tmr.send('mesh_and_space')

# build forms and integrators
bform = BilinearForm(space)
bform.add_integrator(ScalarDiffusionIntegrator())
lform = LinearForm(space)
lform.add_integrator(ScalarSourceIntegrator(source))
tmr.send('forms')

# assembly
A = bform.assembly()
F = lform.assembly()
tmr.send('assembly')
del bform, lform

# apply dirichlet bc
uh = space.function()
A, F = DirichletBC(space, solution).apply(A, F, uh)
tmr.send('dirichlet')

# solve
if CONTEXT == 'torch':
    A = A.to_sparse_csr()
    uh = cg(A, F, uh, maxiter=5000, atol=1e-12, rtol=1e-8).numpy()

elif CONTEXT == 'numpy':
    uh = cg(A, F, uh, maxiter=5000, tol=1e-8, atol=1e-12)

tmr.send('solve')
next(tmr)
del A, F

# fig = plt.figure()
# axes = fig.add_subplot(111, projection='3d')
# mesh_numpy.show_function(axes, uh, cmap='jet')
# plt.show()
