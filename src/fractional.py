
from typing import Dict, Optional, Callable, Sequence

from numpy.typing import NDArray
import torch
from torch.nn import Parameter, Module
from torch import Tensor, float64, device


class Fractional(Module):
    def __init__(self, s: float, w: Tensor, V: Tensor, M: Optional[Tensor]=None,
                 *, dtype=float64, device: device=None) -> None:
        """
        @brief Fractional order operator

        @param s: float. The order of the operator, a parameter of the module.
        @param w: A Tensor containing eigen values of the operator.
        @param V: A tensor containing eigen functions of the operator.
        @param M: The mass matrix, optional.
        """
        super().__init__()
        assert w.ndim == 1
        assert V.ndim == 2
        self.s = Parameter(torch.tensor(s, dtype=dtype, device=device))
        self.w = Parameter(w.to(device=device, dtype=dtype), requires_grad=False)
        self.V = Parameter(V.to(device=device, dtype=dtype), requires_grad=False)
        if M is None:
            Vinv = self.V.T
        else:
            M = M.to(device=device, dtype=dtype)
            Vinv = self.V.T@M
        self.Vinv = Parameter(Vinv, requires_grad=False)

    @classmethod
    def empty(cls, ndofs: int, *, dtype=float64, device: device=None):
        """
        @brief Create an empty fractional operator.

        @param ndofs: int. The number of degrees of freedom.
        """
        s = 0.0
        w = torch.empty((ndofs, ), dtype=dtype, device=device)
        V = torch.empty((ndofs, ndofs), dtype=dtype, device=device)
        return cls(s, w, V, dtype=dtype, device=device)

    @classmethod
    def from_npz(cls, filename: str, s: float, *, dtype=float64, device: device=None):
        """
        @brief Load a fractional operator from a .npz file.

        @param filename: str. The name of the file. The file should contain the following keys:
            - 'w': A 1D tensor containing the eigen values.
            - 'v': A 2D tensor containing the eigen functions.
            - 'M': A 2D tensor containing the mass matrix, optional.

        @param s: float. The order of the operator.

        @return: Fractional. The fractional operator.
        """
        import numpy as np
        data: Dict[str, NDArray] = dict(np.load(filename))
        t_data = {k: torch.from_numpy(v) for k, v in data.items()}

        try:
            if 'M' in t_data:
                return cls(s, t_data['w'], t_data['v'], t_data['M'],
                        dtype=dtype, device=device)
            else:
                return cls(s, t_data['w'], t_data['v'],
                        dtype=dtype, device=device)
        except KeyError:
            raise KeyError(f"The file '{filename}' does not contain the required data.")

    def matrix(self):
        V = self.V
        Vinv = self.Vinv
        L = torch.diag(torch.pow(self.w, self.s))
        return V@L@Vinv

    __call__: Callable[[Tensor], Tensor]

    def forward(self, gdvn: Tensor):
        return self.matrix() @ gdvn


class MultiFractional(Module):
    def __init__(self, s: Sequence[float], w: Tensor, V: Tensor, M: Optional[Tensor]=None,
                 *, dtype=float64, device: device=None) -> None:
        super().__init__()
        assert w.ndim == 1
        assert V.ndim == 2
        self.s = Parameter(torch.tensor(s, dtype=dtype, device=device))
        self.w = Parameter(w.to(device=device, dtype=dtype), requires_grad=False)
        self.V = Parameter(V.to(device=device, dtype=dtype), requires_grad=False)
        if M is None:
            Vinv = self.V.T
        else:
            M = M.to(device=device, dtype=dtype)
            Vinv = self.V.T@M
        self.Vinv = Parameter(Vinv, requires_grad=False)

    @classmethod
    def empty(cls, n_channel: int, ndofs: int, *, dtype=float64, device: device=None):
        """
        @brief Create an empty fractional operator.

        @param ndofs: int. The number of degrees of freedom.
        """
        s = torch.zeros((n_channel, ), dtype=dtype, device=device)
        w = torch.empty((ndofs, ), dtype=dtype, device=device)
        V = torch.empty((ndofs, ndofs), dtype=dtype, device=device)
        return cls(s, w, V, dtype=dtype, device=device)

    @classmethod
    def from_npz(cls, filename: str, s: Sequence[float], *,
                 dtype=float64, device: device=None):
        """
        @brief Load a fractional operator from a .npz file.

        @param filename: str. The name of the file. The file should contain the following keys:
            - 'w': A 1D tensor containing the eigen values.
            - 'v': A 2D tensor containing the eigen functions.
            - 'M': A 2D tensor containing the mass matrix, optional.

        @param s: float. The order of the operator.

        @return: Fractional. The fractional operator.
        """
        import numpy as np
        data: Dict[str, NDArray] = dict(np.load(filename))
        t_data = {k: torch.from_numpy(v) for k, v in data.items()}

        try:
            if 'M' in t_data:
                return cls(s, t_data['w'], t_data['v'], t_data['M'],
                        dtype=dtype, device=device)
            else:
                return cls(s, t_data['w'], t_data['v'],
                        dtype=dtype, device=device)
        except KeyError:
            raise KeyError(f"The file '{filename}' does not contain the required data.")

    def matrix(self): # -> [n_channel, n_dof, n_dof]
        V = self.V
        Vinv = self.Vinv
        L = torch.pow(self.w[None, :], self.s[:, None])
        return torch.einsum('ij, cj, jk -> cik', V, L, Vinv)

    __call__: Callable[[Tensor], Tensor]

    def forward(self, multi_channel_uh: Tensor): # [n_channel, n_dof] -> [n_channel, n_dof]
        return torch.einsum('cik, ck -> ci', self.matrix(), multi_channel_uh)
