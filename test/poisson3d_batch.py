
from fealpy.backend import backend_manager as bm
from fealpy.pde.poisson_3d import BatchedCosCosCosData
from fealpy.mesh import TetrahedronMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import (
    BilinearForm, LinearForm,
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    ScalarNeumannBCIntegrator
)
from fealpy.solver import cg
from fealpy.utils import timer
from fealpy import logger

logger.setLevel('INFO')
bm.set_backend('pytorch')
tmr = timer()

if bm.backend_name == 'pytorch':
    bm.set_default_device('cuda:4')

for _ in range(2):

    next(tmr)
    OMEGA = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pde = BatchedCosCosCosData(omega=OMEGA, dtype=bm.float64)
    mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=32, ny=32, nz=32)
    space = LagrangeFESpace(mesh, p=1)

    bform = BilinearForm(space)
    bform.add_integrator(ScalarDiffusionIntegrator(method='fast'))
    A = bform.assembly()

    lform = LinearForm(space, batch_size=len(OMEGA))
    lform.add_integrator(ScalarSourceIntegrator(pde.source, batched=True))
    lform.add_integrator(ScalarNeumannBCIntegrator(pde.neumann, batched=True))
    F = lform.assembly()

    uh = cg(A, F, batch_first=True, atol=1e-12, rtol=0.0)

    tmr.send('solve')

next(tmr)

# 64*64*64, p=1
"""
[10-28 13:45:34][INFO] fealpy: Mesh toplogy relation constructed, with 1572864 cells, 3170304 faces, 274625 nodes on device ?
[10-28 13:45:34][INFO] fealpy: Bilinear form matrix constructed, with shape [274625, 274625].
[10-28 13:45:34][INFO] fealpy: Linear form vector constructed, with shape [10, 274625].
[10-28 13:45:35][INFO] fealpy: CG: converged in 340 iterations, stopped by absolute tolerance.
Timer received None and paused.
=================================================
   ID       Time        Proportion(%)    Label
-------------------------------------------------
    1      1.599 [s]          100.000    solve
=================================================
"""

### 64*64*64, p=2
"""
[10-28 13:48:42][INFO] fealpy: Mesh toplogy relation constructed, with 1572864 cells, 3170304 faces, 274625 nodes on device ?
[10-28 13:48:43][INFO] fealpy: Bilinear form matrix constructed, with shape [2146689, 2146689].
[10-28 13:48:43][INFO] fealpy: Linear form vector constructed, with shape [10, 2146689].
[10-28 13:48:51][INFO] fealpy: CG: converged in 623 iterations, stopped by absolute tolerance.
Timer received None and paused.
=================================================
   ID       Time        Proportion(%)    Label
-------------------------------------------------
    1      8.895 [s]          100.000    solve
=================================================
"""

### 32*32*32, p=1
"""
[10-28 13:53:26][INFO] fealpy: Mesh toplogy relation constructed, with 196608 cells, 399360 faces, 35937 nodes on device ?
[10-28 13:53:26][INFO] fealpy: Bilinear form matrix constructed, with shape [35937, 35937].
[10-28 13:53:26][INFO] fealpy: Linear form vector constructed, with shape [10, 35937].
[10-28 13:53:26][INFO] fealpy: CG: converged in 188 iterations, stopped by absolute tolerance.
Timer received None and paused.
=================================================
   ID       Time        Proportion(%)    Label
-------------------------------------------------
    1    145.584 [ms]         100.000    solve
=================================================
"""