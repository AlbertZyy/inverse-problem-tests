
from typing import Callable
import torch
from torch import Tensor
from torch.nn import Module

from fdm import LaplaceFDMSolver
from fractional import Fractional, MultiChannelFractional


class DataFeatureFDMSolver(Module):
    def __init__(self, laplace_solver: LaplaceFDMSolver, fractional_lb: Fractional) -> None:
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
    def __init__(self, laplace_solver: LaplaceFDMSolver, fractional_lb: MultiChannelFractional) -> None:
        super().__init__()
        self._solver = laplace_solver
        self._frac = fractional_lb

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

    def forward(self, data: Tensor):
        assert data.ndim == 4
        BATCH_CHANNEL = data.shape[0:2]
        GDGN = data.shape[2:]
        data = data.reshape(-1, *GDGN)
        val = self.solve_phi(data)
        MESH = self._solver.indexing.shape
        return val.reshape(BATCH_CHANNEL + MESH)
