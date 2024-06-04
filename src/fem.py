
import torch
from torch import Tensor, cat
from torch.linalg import lu_factor, lu_solve

from fealpy.torch.mesh import TriangleMesh
from fealpy.torch.functionspace import LagrangeFESpace
from fealpy.torch.fem import BilinearForm, LinearForm
from fealpy.torch.fem import (
    ScalarDiffusionIntegrator,
    ScalarBoundarySourceIntegrator,
    DirichletBC
)


class LaplaceFEMSolver():
    def __init__(self, mesh: TriangleMesh) -> None:
        self.space = LagrangeFESpace(mesh, p=1)
        self.dtype = mesh.ftype
        self.device = mesh.device

        bform = BilinearForm(self.space)
        bform.add_domain_integrator(ScalarDiffusionIntegrator())
        self._A = bform.assembly()
        self.dbc = DirichletBC(self.space)
        self.bsi = ScalarBoundarySourceIntegrator(None, batched=True)

        self.bd_dof = self.space.is_boundary_dof()
        self.ips = self.space.interpolation_points()
        bd_face = mesh.ds.boundary_face_index()
        self.bd_face2dof = self.space.face_to_dof()[bd_face, :]

    def _init_gd(self):
        A_d = self.dbc.apply_matrix(self._A).to_dense()
        self.A_d_LU, self.pivots_d = lu_factor(A_d)

    def _init_gn(self):
        lform_c = LinearForm(self.space)
        lform_c.add_integrator(
            ScalarBoundarySourceIntegrator(1.)
        )
        c = lform_c.assembly()[None, :]
        A = self._A.to_dense()
        ZERO = torch.zeros((1, 1), dtype=self.dtype, device=self.device)
        A_n = cat([
            cat([A, c.T], dim=1),
            cat([c, ZERO], dim=1)
        ], dim=0)
        self.A_n_LU, self.pivots_n = lu_factor(A_n)

    def solve_from_gd(self, gd: Tensor) -> Tensor:
        """Solve Laplace equation with Dirichlet boundary condition.

        Args:
            gd (Tensor): Dirichlet condition on **boundary Dofs**. Requires batch\
                on the first dimension.

        Returns:
            Tensor: A tensor shaped (batch_size, gdof).
        """
        if not hasattr(self, 'A_d_LU'):
            self._init_gd()

        lform = LinearForm(self.space, batch_size=gd.shape[0])
        f = lform.assembly() # An zero tensor
        f = self.dbc.apply_vector(f, self._A, gd=gd)
        uh = lu_solve(self.A_d_LU, self.pivots_d, f, left=True)
        return uh

    def solve_from_gn(self, gn: Tensor) -> Tensor:
        """Solve Laplace equation with Neumann boundary condition.

        Args:
            gn (Tensor): Neumann condition on **boundary face quadrature points**.\
                Requires batch on the first dimension.

        Returns:
            Tensor: A tensor shaped (batch_size, gdof).
        """
        if not hasattr(self, 'A_n_LU'):
            self._init_gn()

        lform = LinearForm(self.space, batch_size=gn.shape[0])
        self.bsi.f = gn
        self.bsi.clear(result_only=True)
        lform.add_integrator(self.bsi)
        f = lform.assembly()
        uh = lu_solve(self.A_n_LU, self.pivots_n, f, left=True)
        return uh

    def boundary_value(self, uh: Tensor) -> Tensor:
        """Find values on boundary interpolation points.

        Args:
            uh (Tensor): Dofs.

        Returns:
            Tensor: A 1-d tensor sized the number of boundary interpolation points.
        """
        return uh[..., self.bd_dof]

    def normal_derivative(self, uh: Tensor) -> Tensor:
        """Calculate normal derivatives on boundary face quadrature points.

        Args:
            uh (Tensor): Dofs.

        Returns:
            Tensor: A 2-d tensor shaped [number of boundary faces, number of\
                local quadrature points].
        """
        bcs, ws, _, fm, index = self.bsi.fetch(self.space)
        nd = uh[..., self.bd_face2dof]
