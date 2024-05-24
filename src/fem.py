
from fealpy.torch.mesh import TriangleMesh
from fealpy.torch.functionspace import LagrangeFESpace
from fealpy.torch.fem import BilinearForm, LinearForm

from torch.linalg import lu_factor


class LaplaceFEMSolver():
    def __init__(self, mesh: TriangleMesh) -> None:
        self.space = LagrangeFESpace(mesh, p=1)
        self.dtype = mesh.ftype
        self.device = mesh.device

    def _init_gd(self):
        from fealpy.torch.fem import ScalarDiffusionIntegrator
        bform = BilinearForm(self.space)
        bform.add_domain_integrator(ScalarDiffusionIntegrator())
        A_d = bform.assembly().to_dense()
        self.A_d_LU, self.pivots_d = lu_factor(A_d)
