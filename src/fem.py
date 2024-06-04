
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
    def __init__(self, mesh: TriangleMesh, p: int=1) -> None:
        self.space = LagrangeFESpace(mesh, p=p)
        self.dtype = mesh.ftype
        self.device = mesh.device

        bform = BilinearForm(self.space)
        bform.add_integrator(ScalarDiffusionIntegrator())
        self._A = bform.assembly()
        self.dbc = DirichletBC(self.space)
        self.bsi = ScalarBoundarySourceIntegrator(None, zero_integral=True, batched=True)

        self.bd_dof_flag = self.space.is_boundary_dof()
        self.bd_cell = mesh.ds.boundary_cell_index()
        self.ips = self.space.interpolation_points()
        bd_face = mesh.ds.boundary_face_index()
        self.cell2dof = self.space.cell_to_dof()
        self.bd_en = mesh.edge_unit_normal(index=bd_face)

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

    def solve_from_gd(self, gd: Tensor, batch_size: int=0) -> Tensor:
        """Solve Laplace equation with Dirichlet boundary condition.

        Args:
            gd (Tensor): Dirichlet condition on **boundary Dofs**. Requires batch\
                on the first dimension.
            batch_size (int, optional): Batch size. Defaults to 0.

        Returns:
            Tensor: A tensor shaped (batch_size, gdof).
        """
        if not hasattr(self, 'A_d_LU'):
            self._init_gd()

        lform = LinearForm(self.space, batch_size=batch_size)
        f = lform.assembly() # An zero tensor
        f = self.dbc.apply_vector(f, self._A, gd=gd)
        uh = lu_solve(self.A_d_LU, self.pivots_d, f, left=False)
        return uh

    def solve_from_gn(self, gn: Tensor, batch_size: int=0) -> Tensor:
        """Solve Laplace equation with Neumann boundary condition.

        Args:
            gn (Tensor): Neumann condition on **boundary face quadrature points**.\
                Requires batch on the first dimension.
            batch_size (int, optional): Batch size. Defaults to 0.

        Returns:
            Tensor: A tensor shaped (batch_size, gdof).
        """
        if not hasattr(self, 'A_n_LU'):
            self._init_gn()

        self.bsi.f = gn
        self.bsi.clear(result_only=True)
        lform = LinearForm(self.space, batch_size=batch_size)
        lform.add_integrator(self.bsi)
        f = lform.assembly()

        if f.ndim > 1:
            ZERO = torch.zeros((f.shape[0], 1), dtype=self.dtype, device=self.device)
            f = torch.cat([f, ZERO], dim=-1)
        elif f.ndim == 1:
            ZERO = torch.zeros((1, ), dtype=self.dtype, device=self.device)
            f = torch.cat([f, ZERO], dim=-1)
        else:
            raise RuntimeError(f"Invalid f.ndim {f.ndim}.")

        uh = lu_solve(self.A_n_LU, self.pivots_n, f, left=False)
        return uh[..., :-1]

    def boundary_value(self, uh: Tensor) -> Tensor:
        """Find values on boundary interpolation points.

        Args:
            uh (Tensor): Dofs.

        Returns:
            Tensor: A 1-d tensor sized the number of boundary interpolation points.
        """
        return uh[..., self.bd_dof_flag]

    def normal_derivative(self, uh: Tensor) -> Tensor:
        """Calculate normal derivatives on boundary face quadrature points.

        Args:
            uh (Tensor): Dofs.

        Returns:
            Tensor: A 2-d tensor shaped [number of boundary faces, number of\
                local quadrature points].
        """
        bcs, ws, _, fm, index = self.bsi.fetch(self.space)

        # extend face bcs to cell bcs
        bd_face2cell = self.space.mesh.ds.face2cell[index, :]
        bd_local_idx = bd_face2cell[:, 2]
        bd_left_cell = bd_face2cell[:, 0]
        bd_left_cell_dof = self.cell2dof[bd_left_cell]

        if uh.ndim > 1:
            result_shape = (uh.shape[0], ws.shape[0], fm.shape[0]) # (B, Q, bd_F)
        else:
            result_shape = (ws.shape[0], fm.shape[0]) # (Q, bd_F)

        result = torch.zeros(result_shape, dtype=self.dtype, device=self.device)
        bd_uh = uh[..., bd_left_cell_dof] # (B, bd_F, I3)

        for i in range(3):
            sub_idx = (bd_local_idx == i).nonzero(as_tuple=True)[0] # 边界边中局部编号是 i 的边
            if sub_idx.numel() == 0:
                continue

            ZERO = torch.zeros((bcs.shape[0], 1), dtype=bcs.dtype, device=bcs.device)
            new_bcs = torch.cat([bcs[:, :i], ZERO, bcs[:, i:]], dim=1)
            if i == 1:
                new_bcs = new_bcs.flip(dims=(1,))
            assert new_bcs.shape == (bcs.shape[0], 3)

            sub_uh = bd_uh[..., sub_idx, :] # (B, sub_F, I3)
            sub_lcell = bd_left_cell[sub_idx] # 边界边中局部编号是 i 的边 的 左边单元全局编号
            sub_gphi = self.space.grad_basis(new_bcs, index=sub_lcell, variable='x') # (Q, sub_F, I3, GD)
            sub_en = self.bd_en[sub_idx]
            nd = torch.einsum('fm, qfim, ...fi -> ...qf', sub_en, sub_gphi, sub_uh) # (B, Q, sub_F)

            result[..., sub_idx] = nd

        return result
