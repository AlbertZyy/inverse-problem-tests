from time import time

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

for i in range(LOOPS + 1): # warm up
    t0 = time()
    OMEGA = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    results = []

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
        ZERO = bm.zeros((1,), dtype=F.dtype, device=F.device)
        F = bm.concat([F, ZERO], axis=0)

        uh = cg(A, F, atol=1e-12, rtol=0.0)
        results.append(uh[:-1])

    uh = bm.stack(results, axis=0)
    t1 = time()

    if i > 1:
        total_time += t1 - t0

print(total_time / LOOPS)


# 32*32*32, p=1, 0.7407291650772094
# 64*64*64, p=1, 1.632502293586731
# 64*64*64, p=2, 7.014384746551514