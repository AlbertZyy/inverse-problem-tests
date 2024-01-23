
from typing import Dict, Optional, Callable, Sequence

from numpy.typing import NDArray
import torch
from torch.nn import Parameter, Module, init
from torch import Tensor, float64, device, relu


class Fractional(Module):
    def __init__(self, n_dofs: int, *, dtype=float64, device: device=None) -> None:
        super().__init__()
        kwargs = dict(dtype=dtype, device=device)
        self.n_dofs = n_dofs
        self.s = Parameter(torch.empty((), **kwargs))
        self.w = Parameter(torch.empty((n_dofs, ), **kwargs), requires_grad=False)
        self.V = Parameter(torch.empty((n_dofs, n_dofs), **kwargs), requires_grad=False)
        self.Vinv = Parameter(torch.empty((n_dofs, n_dofs), **kwargs), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.s, 0.0)
        init.zeros_(self.w)
        init.orthogonal_(self.V)
        with torch.no_grad():
            # NOTE: Data should be copied from V.T to Vinv. Otherwise, V will be
            # overriten by Vinv when loading the state dict.
            self.Vinv.copy_(self.V.T)

    def setup(self, w: Tensor, V: Tensor, Vinv: Optional[Tensor]=None):
        assert w.ndim == 1
        assert V.ndim == 2
        with torch.no_grad():
            self.w.copy_(w)
            self.V.copy_(V)
            if Vinv is None:
                Vinv = self.V.T
            self.Vinv.copy_(Vinv)

    def initialize(self, s: float):
        """
        @brief Initialize the order of the fractional operator.
        """
        with torch.no_grad():
            init.constant_(self.s, s)

    def from_npz(self, filename: str):
        """
        @brief Load a fractional operator from a .npz file.

        @param filename: str. The name of the file. The file may contain the following keys:
            - 'w': A 1D tensor containing the eigen values.
            - 'v': A 2D tensor containing the eigen functions.
            - 'vinv': A 2D tensor containing the inverse of v, optional.
            - 'M': The 2D mass matrix, satisfying `vinv=v.T@M`, optional. Ignored if `vinv` is provided.

        @param s: float. The order of the operator.
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

    def matrix(self):
        V = self.V
        Vinv = self.Vinv
        L = torch.diag(torch.pow(self.w, self.s))
        return V@L@Vinv

    __call__: Callable[[Tensor], Tensor]

    def forward(self, gdvn: Tensor):
        return torch.einsum('ik, ...k -> ...i', self.matrix(), gdvn)

    def alpha(self, data: Tensor) -> Tensor:
        """
        @brief

        @param data: Tensor. [n_channel, n_dof]
        """
        return torch.einsum('ik, ...k -> ...i', self.Vinv, data)


class FractionalWithHighcut(Fractional):
    def __init__(self, n_dofs: int, hc_slope=2., *, dtype=float64, device: device=None) -> None:
        super().__init__(n_dofs, dtype=dtype, device=device)
        kwargs = dict(dtype=dtype, device=device)
        self.hc = Parameter(torch.empty((), **kwargs), requires_grad=False)
        self.hc_slope = Parameter(torch.tensor(hc_slope, **kwargs), requires_grad=False)

    def initialize(self, s: float, hc: float):
        """
        @brief Initialize the fractional operator order and the eigen value highcut.
        """
        super().initialize(s)
        with torch.no_grad():
            init.constant_(self.hc, hc)

    def matrix(self):
        V = self.V
        Vinv = self.Vinv
        hc = self.hc
        lam = self.w
        L = torch.diag(torch.pow(lam, self.s) * torch.pow(relu(lam/hc - 1) + 1, -self.hc_slope))
        return V@L@Vinv


class MultiChannelFractional(Module):
    def __init__(self, n_dofs: int, n_channels: int, hc_slope=2., *, dtype=float64, device: device=None) -> None:
        super().__init__()
        assert n_channels > 0
        kwargs = dict(dtype=dtype, device=device)
        self.n_dofs = n_dofs
        self.n_channels = n_channels
        self.s = Parameter(torch.empty((n_channels, ), **kwargs))
        self.hc = Parameter(torch.empty((n_channels, ), **kwargs), requires_grad=False)
        self.hc_slope = Parameter(torch.tensor(hc_slope, dtype=dtype, device=device), requires_grad=False)
        self.w = Parameter(torch.empty((n_dofs, ), **kwargs), requires_grad=False)
        self.V = Parameter(torch.empty((n_dofs, n_dofs), **kwargs), requires_grad=False)
        self.Vinv = Parameter(torch.empty((n_dofs, n_dofs), **kwargs), requires_grad=False)
        self.reset_paramters()

    def reset_paramters(self):
        init.constant_(self.s, 0.0)
        init.constant_(self.hc, 1.0)
        init.zeros_(self.w)
        init.orthogonal_(self.V)
        with torch.no_grad():
            # NOTE: Data should be copied from V.T to Vinv. Otherwise, V will be
            # overriten by Vinv when loading the state dict.
            self.Vinv.copy_(self.V.T)

    def setup(self, w: Tensor, V: Tensor, Vinv: Optional[Tensor]=None):
        assert w.ndim == 1
        assert V.ndim == 2
        with torch.no_grad():
            self.w.copy_(w)
            self.V.copy_(V)
            if Vinv is None:
                Vinv = self.V.T
            assert Vinv.ndim == 2
            self.Vinv.copy_(Vinv)

    def initialize(self, s: Sequence[float], hc: Sequence[float]):
        """
        @brief Initialize the fractional operator order and the eigen value highcut\
               for each channel.
        """
        with torch.no_grad():
            self.s.copy_(torch.tensor(s, dtype=self.s.dtype, device=self.s.device))
            self.hc.copy_(torch.tensor(hc, dtype=self.hc.dtype, device=self.hc.device))

    def from_npz(self, filename: str):
        """
        @brief Load a fractional operator from a .npz file.

        @param filename: str. The name of the file. The file should contain the following keys:
            - 'w': A 1D tensor containing the eigen values.
            - 'v': A 2D tensor containing the eigen functions.
            - 'vinv': A 2D tensor containing the inverse of v, optional.
            - 'M': The 2D mass matrix, satisfying `vinv=v.T@M`, optional. Ignored if `vinv` is provided.

        @return: Fractional. The fractional operator.
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

    def matrix(self): # -> [n_channel, n_dof, n_dof]
        V = self.V
        Vinv = self.Vinv
        lam = self.w[None, :]
        hc = self.hc[:, None]
        slope = self.s[:, None]
        L = torch.pow(lam, slope) * torch.pow(relu(lam/hc - 1) + 1, -self.hc_slope)
        return torch.einsum('ij, cj, jk -> cik', V, L, Vinv)

    __call__: Callable[[Tensor], Tensor]

    def forward(self, data: Tensor) -> Tensor: # [n_channel, n_dof] -> [n_channel, n_dof]
        return torch.einsum('cik, ...ck -> ...ci', self.matrix(), data)

    def alpha(self, data: Tensor) -> Tensor:
        """
        @brief

        @param data: Tensor. [n_channel, n_dof]
        """
        return torch.einsum('ik, ...ck -> ...ci', self.Vinv, data)


class EigenvalueFilter(Module):
    def __init__(self, n_channels: int, n_dofs: int, *, dtype=float64, device: device=None) -> None:
        super().__init__()
        assert n_channels >= 1
        assert n_dofs >= 2
        kwargs = dict(dtype=dtype, device=device)
        self.n_channels = n_channels
        self.n_dofs = n_dofs
        self.V = Parameter(torch.empty((n_dofs, n_dofs), **kwargs), requires_grad=False)
        self.Vinv = Parameter(torch.empty((n_dofs, n_dofs), **kwargs), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.zeros_(self.gain)
        init.orthogonal_(self.V)
        with torch.no_grad():
            self.Vinv.copy_(self.V.T)

    def setup(self, v: Tensor, vinv: Optional[Tensor]=None, *, non_blocking=False):
        kwargs = dict(non_blocking=non_blocking)
        with torch.no_grad():
            self.V.copy_(v, **kwargs)

        if vinv is None:
            self.Vinv.copy_(self.V.T, **kwargs)
        else:
            self.Vinv.copy_(vinv, **kwargs)

    def from_npz(self, filename: str) -> None:
        import numpy as np
        data: Dict[str, NDArray] = dict(np.load(filename))
        t_data = {k: torch.from_numpy(v) for k, v in data.items()}
        del data

        try:
            if 'vinv' in t_data:
                self.setup(t_data['v'], t_data['vinv'])
            elif 'M' in t_data:
                Vinv = t_data['v'].T @ t_data['M']
                self.setup(t_data['v'], Vinv)
            else:
                self.setup(t_data['v'])
        except KeyError:
            raise KeyError(f"The file '{filename}' does not contain the required data.")

    __call__: Callable[[Tensor], Tensor]

    def inverse(self, __eigenfunc_coef: Tensor) -> Tensor:
        """
        @brief Map from the eigenfunction domain.
        """
        return torch.einsum('ik, ...ck -> ...ci', self.V, __eigenfunc_coef)

    def direct(self, __func_data: Tensor) -> Tensor:
        """
        @brief Map to the eigenfunction domain.
        """
        return torch.einsum('ik, ...ck -> ...ci', self.Vinv, __func_data)

    forward = direct
