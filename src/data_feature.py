
from typing import Callable, Sequence, Optional
import torch
from torch import device, int32, float64
from torch.nn import Module

from fdm import LaplaceFDMSolver

Tensor = torch.Tensor


class DataFeatureFDMSolver(Module):
    def __init__(self, laplace_solver: LaplaceFDMSolver, fractional_lb: Module) -> None:
        super().__init__()
        self._solver = laplace_solver
        self._frac = fractional_lb
        self._vmapped = torch.vmap(self.solve_phi, 0, 0)

    def solve_phi(self, gd_gn: Tensor) -> Tensor:
        """
        @brief From gd_gn data to data feature.

        @param gd_gn: Tensor in shape (2, NN_bd), where NN_bd is the number of\
               nodes on the boundary.

        @return: Tensor in shape (NN, ).
        """
        gd, gn = gd_gn[0], gd_gn[1]
        vuh = self._solver.solve_from_gd(gd)
        vn = self._solver.normal_derivative(vuh)
        gnvn = self._frac(gn - vn)
        return self._solver.solve_from_gn(gnvn)

    def restore_size(self, phi: Tensor):
        shape = self._solver.indexing.shape
        return phi.reshape(shape)

    __call__: Callable[[Tensor], Tensor]

    # NOTE: data is in shape (N, C, 2, 4*NN), where N is batch, C is channel.
    # Batch is for different sigma levelsets, and channel is for different boundary data.
    # The (2, 4*NN) is the shape of gd_gn.
    def forward(self, data: Tensor):
        assert data.ndim == 4
        BATCH_CHANNEL = data.shape[0:2]
        GDGN = data.shape[2:]
        data = data.reshape(-1, *GDGN)
        val = self._vmapped(data) # (N*C, NN)
        MESH = self._solver.indexing.shape
        return val.reshape(BATCH_CHANNEL + MESH) # (N, C, Nx, Ny)


class MultiChannelDataFeature(Module):
    def __init__(self, laplace_solver: LaplaceFDMSolver, frac_lb: Module) -> None:
        super().__init__()
        self._solver = laplace_solver
        self._frac = frac_lb

        self._frac_input_buffer: Optional[Tensor] = None

    @classmethod
    def from_domain(cls, size: Sequence[int], h: Sequence[float], frac_lb: Module,
                    *, itype=int32, dtype=float64, device: device=None):
        solver = LaplaceFDMSolver(size, h, itype=itype, dtype=dtype, device=device)
        return cls(solver, frac_lb)

    def solve_phi(self, gd_gn: Tensor) -> Tensor:
        """
        @brief From gd_gn data to data feature.

        @param gd_gn: Tensor in shape (C, 2, NN_bd), where NN_bd is the number of\
               nodes on the boundary.

        @return: Tensor in shape (C, NN).
        """
        assert gd_gn.ndim == 3
        assert gd_gn.shape[1] == 2
        gd, gn = gd_gn[:, 0, :], gd_gn[:, 1, :]
        vuh = self._solver.solve_from_gd(gd) # [C, NN]
        vn = self._solver.normal_derivative(vuh) # [C, NN_bd]
        gnvn = self._frac(gn - vn) # [C, NN_bd]
        return self._solver.solve_from_gn(gnvn) # [C, NN]

    __call__: Callable[[Tensor], Tensor]

    def forward(self, data: Tensor):
        assert data.ndim == 4
        assert data.shape[2] == 2
        BATCH, CHANNEL, _, NNBD = data.shape

        # NOTE: Merge the Batch and Channel axis to vectorize in the FDM solver.
        data = data.reshape(-1, 2, NNBD) # [N*C, 2, NN_bd]
        gd, gn = data[:, 0, :], data[:, 1, :] # [N*C, NN_bd]
        vuh = self._solver.solve_from_gd(gd) # [N*C, NN]
        vn = self._solver.normal_derivative(vuh) # [N*C, NN_bd]

        # NOTE: Restore the Batch axis to send to the fractional.
        _frac_input_buffer = (gn - vn).reshape(BATCH, CHANNEL, NNBD)
        gnvn = self._frac(_frac_input_buffer) # [N, C, NN_bd]
        self._frac_input_buffer = _frac_input_buffer

        # NOTE: Merge the Batch and Channel axis again for the FDM solver.
        val = self._solver.solve_from_gn(gnvn.reshape(-1, NNBD)) # [N*C, NN]

        # NOTE: Return the result as a 4-d image.
        MESH = self._solver.indexing.shape
        return val.reshape((BATCH, CHANNEL) + MESH)

    def frac_input(self):
        return self._frac_input_buffer

    # NOTE: This method is for testing, do not call it in other methods or in training.
    def gd2gn_diff(self, data: Tensor):
        assert data.ndim == 4
        assert data.shape[2] == 2
        BATCH, CHANNEL, _, NNBD = data.shape

        # NOTE: Merge the Batch and Channel axis to vectorize in the FDM solver.
        data = data.reshape(-1, 2, NNBD) # [N*C, 2, NN_bd]
        gd, gn = data[:, 0, :], data[:, 1, :] # [N*C, NN_bd]
        vuh = self._solver.solve_from_gd(gd) # [N*C, NN]
        vn = self._solver.normal_derivative(vuh) # [N*C, NN_bd]

        # NOTE: Restore the Batch axis to return.
        return (gn - vn).reshape(BATCH, CHANNEL, NNBD) # [N, C, NN_bd]
