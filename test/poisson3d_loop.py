
from fealpy.backend import backend_manager as bm
from fealpy.pde.poisson_3d import CosCosCosData
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

for _ in range(2): # warm up

    next(tmr)
    OMEGA = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    results = []

    mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=32, ny=32, nz=32)
    space = LagrangeFESpace(mesh, p=1)

    bform = BilinearForm(space)
    bform.add_integrator(ScalarDiffusionIntegrator(method='fast'))
    A = bform.assembly()

    si = ScalarSourceIntegrator()
    bci = ScalarNeumannBCIntegrator()

    for omega in OMEGA:
        pde = CosCosCosData(omega=omega)
        si.source = pde.source
        si.clear()
        bci.gn = pde.neumann
        bci.clear()

        lform = LinearForm(space)
        lform.add_integrator(si)
        lform.add_integrator(bci)
        F = lform.assembly()

        uh = cg(A, F, atol=1e-12, rtol=0.0)


        results.append(uh)

    uh = bm.stack(results, axis=0)
    tmr.send('solve')

next(tmr)

# 64*64*64, p=1
"""
[10-28 13:43:35][INFO] fealpy: Mesh toplogy relation constructed, with 1572864 cells, 3170304 faces, 274625 nodes on device ?
[10-28 13:43:35][INFO] fealpy: Bilinear form matrix constructed, with shape [274625, 274625].
[10-28 13:43:35][INFO] fealpy: Linear form vector constructed, with shape [274625].
[10-28 13:43:35][INFO] fealpy: CG: converged in 315 iterations, stopped by absolute tolerance.
[10-28 13:43:35][INFO] fealpy: Linear form vector constructed, with shape [274625].
[10-28 13:43:36][INFO] fealpy: CG: converged in 323 iterations, stopped by absolute tolerance.
[10-28 13:43:36][INFO] fealpy: Linear form vector constructed, with shape [274625].
[10-28 13:43:36][INFO] fealpy: CG: converged in 334 iterations, stopped by absolute tolerance.
[10-28 13:43:36][INFO] fealpy: Linear form vector constructed, with shape [274625].
[10-28 13:43:36][INFO] fealpy: CG: converged in 330 iterations, stopped by absolute tolerance.
[10-28 13:43:36][INFO] fealpy: Linear form vector constructed, with shape [274625].
[10-28 13:43:36][INFO] fealpy: CG: converged in 338 iterations, stopped by absolute tolerance.
[10-28 13:43:36][INFO] fealpy: Linear form vector constructed, with shape [274625].
[10-28 13:43:36][INFO] fealpy: CG: converged in 327 iterations, stopped by absolute tolerance.
[10-28 13:43:36][INFO] fealpy: Linear form vector constructed, with shape [274625].
[10-28 13:43:36][INFO] fealpy: CG: converged in 332 iterations, stopped by absolute tolerance.
[10-28 13:43:36][INFO] fealpy: Linear form vector constructed, with shape [274625].
[10-28 13:43:36][INFO] fealpy: CG: converged in 324 iterations, stopped by absolute tolerance.
[10-28 13:43:36][INFO] fealpy: Linear form vector constructed, with shape [274625].
[10-28 13:43:37][INFO] fealpy: CG: converged in 331 iterations, stopped by absolute tolerance.
[10-28 13:43:37][INFO] fealpy: Linear form vector constructed, with shape [274625].
[10-28 13:43:37][INFO] fealpy: CG: converged in 322 iterations, stopped by absolute tolerance.
Timer received None and paused.
=================================================
   ID       Time        Proportion(%)    Label
-------------------------------------------------
    1    628.980 [ms]         100.000    solve
=================================================
"""

### 64*64*64, p=2
"""
[10-28 13:50:47][INFO] fealpy: Mesh toplogy relation constructed, with 1572864 cells, 3170304 faces, 274625 nodes on device ?
[10-28 13:50:47][INFO] fealpy: Bilinear form matrix constructed, with shape [2146689, 2146689].
[10-28 13:50:47][INFO] fealpy: Linear form vector constructed, with shape [2146689].
[10-28 13:50:48][INFO] fealpy: CG: converged in 546 iterations, stopped by absolute tolerance.
[10-28 13:50:48][INFO] fealpy: Linear form vector constructed, with shape [2146689].
[10-28 13:50:48][INFO] fealpy: CG: converged in 578 iterations, stopped by absolute tolerance.
[10-28 13:50:48][INFO] fealpy: Linear form vector constructed, with shape [2146689].
[10-28 13:50:49][INFO] fealpy: CG: converged in 614 iterations, stopped by absolute tolerance.
[10-28 13:50:49][INFO] fealpy: Linear form vector constructed, with shape [2146689].
[10-28 13:50:50][INFO] fealpy: CG: converged in 573 iterations, stopped by absolute tolerance.
[10-28 13:50:50][INFO] fealpy: Linear form vector constructed, with shape [2146689].
[10-28 13:50:51][INFO] fealpy: CG: converged in 583 iterations, stopped by absolute tolerance.
[10-28 13:50:51][INFO] fealpy: Linear form vector constructed, with shape [2146689].
[10-28 13:50:51][INFO] fealpy: CG: converged in 579 iterations, stopped by absolute tolerance.
[10-28 13:50:51][INFO] fealpy: Linear form vector constructed, with shape [2146689].
[10-28 13:50:52][INFO] fealpy: CG: converged in 568 iterations, stopped by absolute tolerance.
[10-28 13:50:52][INFO] fealpy: Linear form vector constructed, with shape [2146689].
[10-28 13:50:53][INFO] fealpy: CG: converged in 582 iterations, stopped by absolute tolerance.
[10-28 13:50:53][INFO] fealpy: Linear form vector constructed, with shape [2146689].
[10-28 13:50:53][INFO] fealpy: CG: converged in 573 iterations, stopped by absolute tolerance.
[10-28 13:50:53][INFO] fealpy: Linear form vector constructed, with shape [2146689].
[10-28 13:50:54][INFO] fealpy: CG: converged in 597 iterations, stopped by absolute tolerance.
Timer received None and paused.
=================================================
   ID       Time        Proportion(%)    Label
-------------------------------------------------
    1      7.299 [s]          100.000    solve
=================================================
"""

### 32*32*32, p=1
"""
[10-28 13:54:07][INFO] fealpy: Mesh toplogy relation constructed, with 196608 cells, 399360 faces, 35937 nodes on device ?
[10-28 13:54:07][INFO] fealpy: Bilinear form matrix constructed, with shape [35937, 35937].
[10-28 13:54:07][INFO] fealpy: Linear form vector constructed, with shape [35937].
[10-28 13:54:07][INFO] fealpy: CG: converged in 173 iterations, stopped by absolute tolerance.
[10-28 13:54:07][INFO] fealpy: Linear form vector constructed, with shape [35937].
[10-28 13:54:07][INFO] fealpy: CG: converged in 178 iterations, stopped by absolute tolerance.
[10-28 13:54:07][INFO] fealpy: Linear form vector constructed, with shape [35937].
[10-28 13:54:07][INFO] fealpy: CG: converged in 186 iterations, stopped by absolute tolerance.
[10-28 13:54:07][INFO] fealpy: Linear form vector constructed, with shape [35937].
[10-28 13:54:07][INFO] fealpy: CG: converged in 180 iterations, stopped by absolute tolerance.
[10-28 13:54:07][INFO] fealpy: Linear form vector constructed, with shape [35937].
[10-28 13:54:07][INFO] fealpy: CG: converged in 185 iterations, stopped by absolute tolerance.
[10-28 13:54:07][INFO] fealpy: Linear form vector constructed, with shape [35937].
[10-28 13:54:07][INFO] fealpy: CG: converged in 181 iterations, stopped by absolute tolerance.
[10-28 13:54:07][INFO] fealpy: Linear form vector constructed, with shape [35937].
[10-28 13:54:07][INFO] fealpy: CG: converged in 183 iterations, stopped by absolute tolerance.
[10-28 13:54:07][INFO] fealpy: Linear form vector constructed, with shape [35937].
[10-28 13:54:07][INFO] fealpy: CG: converged in 180 iterations, stopped by absolute tolerance.
[10-28 13:54:07][INFO] fealpy: Linear form vector constructed, with shape [35937].
[10-28 13:54:07][INFO] fealpy: CG: converged in 183 iterations, stopped by absolute tolerance.
[10-28 13:54:07][INFO] fealpy: Linear form vector constructed, with shape [35937].
[10-28 13:54:07][INFO] fealpy: CG: converged in 179 iterations, stopped by absolute tolerance.
Timer received None and paused.
=================================================
   ID       Time        Proportion(%)    Label
-------------------------------------------------
    1    807.079 [ms]         100.000    solve
=================================================
"""