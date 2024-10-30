from time import time

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
from fealpy.sparse import COOTensor
from fealpy.solver import cg
from fealpy.utils import timer

bm.set_backend('pytorch')
tmr = timer()

if bm.backend_name == 'pytorch':
    bm.set_default_device('cuda:7')

total_time = 0.
LOOPS = 10
EXT = 64
P = 2

for i in range(LOOPS + 1):
    t0 = time()
    OMEGA = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pde = BatchedCosCosCosData(omega=OMEGA, dtype=bm.float64)
    mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=EXT, ny=EXT, nz=EXT)
    space = LagrangeFESpace(mesh, p=P)
    GDOF = space.number_of_global_dofs()

    bform = BilinearForm(space)
    bform.add_integrator(ScalarDiffusionIntegrator(method='fast'))
    A = bform.assembly(format='coo')

    lform_c = LinearForm(space)
    lform_c.add_integrator(ScalarNeumannBCIntegrator(1.))
    c = lform_c.assembly(format='coo')

    new_value = bm.concat([A.values(), c.values(), c.values()], axis=0)
    c_idx = bm.concat([c.indices(), bm.full_like(c.indices(), GDOF)], axis=0)
    new_indices = bm.concat(
        [A.indices(), c_idx, bm.flip(c_idx, axis=0)], axis=-1
    )
    A = COOTensor(new_indices, new_value, spshape=(GDOF+1, GDOF+1)).tocsr()

    lform.add_integrator(ScalarSourceIntegrator(pde.source, batched=True))
    lform.add_integrator(ScalarNeumannBCIntegrator(pde.neumann, batched=True))
    F = lform.assembly()
    ZERO = bm.zeros((len(OMEGA), 1), dtype=F.dtype, device=F.device)
    F = bm.concat([F, ZERO], axis=-1)

    lform = LinearForm(space, batch_size=len(OMEGA))
    uh = cg(A, F, batch_first=True, atol=1e-12, rtol=0.0)[:, :-1]
    t1 = time()

    if i > 1:
        total_time += t1 - t0

print(total_time / LOOPS)


# 32*32*32, p=1, 0.15490543842315674
# 64*64*64, p=1, 0.5942094564437866
# 64*64*64, p=2, 7.711729693412781