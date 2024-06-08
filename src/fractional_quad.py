
from typing import Union, Dict, Optional, Callable, Sequence
from math import log

from numpy.typing import NDArray
import torch
from torch.nn import Parameter, Module, init
from torch.nn import functional as F
from torch import float64, device, relu

from fealpy.torch.mesh import Mesh
from fealpy.torch.functionspace import LagrangeFESpace


_dtype = torch.dtype
_device = torch.device
Tensor = torch.Tensor
Index = Union[int, Tensor, Sequence[int], slice]

_S = slice(None, None, None)


class _EigenvalueBase(Module):
    n_dofs: int
    w: Tensor
    V: Tensor
    eigen_func: Tensor

    def __init__(self, n_dofs: int, mesh: Mesh, q: int=3,
                 *, dtype: Optional[_dtype]=float64,
                 device: Union[_device, str, None]=None) -> None:
        super(_EigenvalueBase, self).__init__()
        kwargs = dict(dtype=dtype, device=device)
        self.n_dofs = n_dofs
        self.register_buffer('w', torch.empty((n_dofs, ), **kwargs))
        self.register_buffer('V', torch.empty((n_dofs, n_dofs), **kwargs))
        self.mesh = mesh
        self.space = LagrangeFESpace(mesh, p=1)
        self.cell2dof = self.space.cell_to_dof()
        self.register_buffer(
            'eigen_func', torch.empty((n_dofs, ) + self.cell2dof.shape, **kwargs)
        )
        qf = mesh.integrator(3, 'cell')
        bcs, self.ws = qf.get_quadrature_points_and_weights()
        self.basis = self.space.basis(bcs, variable='x') # (Q, 1, ldof)

    def reset_operator(self):
        init.zeros_(self.w)
        init.orthogonal_(self.V)
        # NOTE: Data should be copied from V.T to Vinv. Otherwise, V will be
        # overriten by Vinv when loading the state dict.
        # self.Vinv.copy_(self.V.T)

    def setup(self, w: Tensor, V: Tensor) -> None:
        assert w.ndim == 1 # (eigen, )
        assert V.ndim == 2 # (gdof, eigen)
        self.w.copy_(w)
        self.V.copy_(V)

        eigen_func = torch.einsum(
            'qci, cie -> eqc',
            self.basis, # (Q, C, ldof)
            V[self.cell2dof, ...] # (C, ldof, eigen)
        ) # (eigen, Q, C)
        self.eigen_func.copy_(eigen_func)

    def from_npz(self, filename: str):
        """
        @brief Load a fractional operator from a .npz file.

        @param filename: str. The name of the file. The file may contain the following keys:
            - 'w': A 1D tensor containing the eigen values.
            - 'v': A 2D tensor containing the eigen functions.
            - 'vinv': A 2D tensor containing the inverse of v, optional.
            - 'M': The 2D mass matrix, satisfying `vinv=v.T@M`, optional. Ignored if `vinv` is provided.
        """
        import numpy as np
        data: Dict[str, NDArray] = dict(np.load(filename))
        t_data = {k: torch.from_numpy(v) for k, v in data.items()}

        try:
            if 'vinv' in t_data:
                self.setup(t_data['w'], t_data['v'], t_data['vinv'])
            elif 'M' in t_data:
                Vinv = t_data['v'].T @ t_data['M']
                self.setup(t_data['w'], t_data['v'], Vinv)
            else:
                self.setup(t_data['w'], t_data['v'])
        except KeyError:
            raise KeyError(f"The file '{filename}' does not contain the required data.")

    def decompose(self, function: Tensor) -> Tensor:
        return torch.einsum('ik, ...k -> ...i', self.Vinv, function)

    def reconstruct(self, eigen_coef: Tensor) -> Tensor:
        return torch.einsum('ik, ...k -> ...i', self.V, eigen_coef)


class Fractional(_EigenvalueBase):
    def __init__(self, n_dofs: int, *, dtype: Optional[_dtype]=float64,
                 device: Union[_device, str, None]=None) -> None:
        super().__init__(n_dofs, dtype=dtype, device=device)
        kwargs = dict(dtype=dtype, device=device)
        self.s = Parameter(torch.zeros((), **kwargs))
        self.reset_operator()

    def initialize(self, s: float):
        """
        @brief Initialize the order of the fractional operator.
        """
        with torch.no_grad():
            init.constant_(self.s, s)

    def matrix(self):
        V = self.V
        Vinv = self.Vinv
        L = torch.diag(torch.pow(self.w, self.s))
        return V@L@Vinv

    __call__: Callable[[Tensor], Tensor]

    def forward(self, gdvn: Tensor):
        return torch.einsum('ik, ...k -> ...i', self.matrix(), gdvn)