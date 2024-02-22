
from typing import Union, Dict, Optional, Callable, Sequence
from math import log

from numpy.typing import NDArray
import torch
from torch.nn import Parameter, Module, init
from torch import Tensor, float64, device, relu


_dtype = torch.dtype
_device = torch.device


class _EigenvalueBase(Module):
    n_dofs: int
    w: Tensor
    V: Tensor
    Vinv: Tensor

    def __init__(self, n_dofs: int, *, dtype: Optional[_dtype]=float64,
                 device: Union[_device, str, None]=None) -> None:
        super(_EigenvalueBase, self).__init__()
        kwargs = dict(dtype=dtype, device=device)
        self.n_dofs = n_dofs
        self.register_buffer('w', torch.empty((n_dofs, ), **kwargs))
        self.register_buffer('V', torch.empty((n_dofs, n_dofs), **kwargs))
        self.register_buffer('Vinv', torch.empty((n_dofs, n_dofs), **kwargs))

    def reset_operator(self):
        init.zeros_(self.w)
        init.orthogonal_(self.V)
        # NOTE: Data should be copied from V.T to Vinv. Otherwise, V will be
        # overriten by Vinv when loading the state dict.
        self.Vinv.copy_(self.V.T)

    def setup(self, w: Tensor, V: Tensor, Vinv: Optional[Tensor]=None):
        assert w.ndim == 1
        assert V.ndim == 2
        self.w.copy_(w)
        self.V.copy_(V)
        if Vinv is None:
            Vinv = self.V.T
        else:
            assert Vinv.ndim == 2
        self.Vinv.copy_(Vinv)

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

    def alpha(self, space_domain: Tensor) -> Tensor:
        return torch.einsum('ik, ...k -> ...i', self.Vinv, space_domain)

    def beta(self, eigen_domain: Tensor) -> Tensor:
        return torch.einsum('ik, ...k -> ...i', self.V, eigen_domain)


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


class MultiChannelFractional(_EigenvalueBase):
    def __init__(self, n_dofs: int, n_channels: int, *,
                 high_cut: bool=False, hc_slope=2.,
                 dtype=float64, device: device=None) -> None:
        super().__init__(n_dofs, dtype=dtype, device=device)
        assert n_channels > 0
        kwargs = dict(dtype=dtype, device=device)
        self.n_channels = n_channels
        self.s = Parameter(torch.empty((n_channels, ), **kwargs))

        if high_cut:
            self.hc = Parameter(torch.empty((n_channels, ), **kwargs), requires_grad=False)
            self.hc_slope = Parameter(torch.tensor(hc_slope, dtype=dtype, device=device), requires_grad=False)
            self.matrix = self._transform_with_high_cut
        else:
            self.register_parameter('hc', None)
            self.register_parameter('hc_slope', None)
            self.matrix = self._transform
        self.reset_operator()
        self.reset_paramters()

    def reset_paramters(self):
        init.constant_(self.s, 0.0)
        if self.hc is not None:
            init.constant_(self.hc, self.w.max().item())

    def initialize(self, s: Sequence[float], hc: Optional[Sequence[float]]=None):
        """
        @brief Initialize the fractional operator order and the eigen value highcut\
               for each channel.
        """
        with torch.no_grad():
            self.s.copy_(torch.tensor(s, dtype=self.s.dtype, device=self.s.device))
            if hc is not None:
                if self.hc is None:
                    raise ValueError("The high cut has been disabled.")
                self.hc.copy_(torch.tensor(hc, dtype=self.hc.dtype, device=self.hc.device))

    def _transform(self):
        V = self.V
        Vinv = self.Vinv
        lam = self.w[None, :]
        slope = self.s[:, None]
        L = torch.pow(lam, slope)
        return torch.einsum('ij, cj, jk -> cik', V, L, Vinv)

    def _transform_with_high_cut(self):
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


class AdaptiveFractional(_EigenvalueBase):
    weights: Tensor
    s: Tensor

    def __init__(self, n_dofs: int, n_channels: int, *,
                 weight: Optional[bool]=True,
                 momentum: float=0.99,
                 eps: float=1e-6,
                 dtype: Optional[_dtype]=float64,
                 device: Union[_device, str, None]=None) -> None:
        super().__init__(n_dofs, dtype=dtype, device=device)
        kwargs = dict(dtype=dtype, device=device)
        self.n_channels = n_channels
        self.enable_weight = weight
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('weights', torch.empty((n_dofs, ), **kwargs))
        self.register_buffer('s', torch.empty((n_channels, ), **kwargs))
        self.reset_operator()
        self.reset_running_stats(weight)

    def setup(self, w: Tensor, V: Tensor, Vinv: Optional[Tensor] = None):
        super().setup(w, V, Vinv)
        self.reset_running_stats(self.enable_weight)

    def reset_running_stats(self, enable_weight: bool):
        self.s.zero_()
        log_eigen = torch.log10(self.w) # [n_dofs, ]
        log_eigen = log_eigen - log_eigen.mean() # scalar

        if enable_weight:
            weight = 1/(self.w*log(10))
            weight.sqrt_()
            W = torch.outer(weight, weight)

            torch.matmul(W, log_eigen, out=self.weights)
            self.weights.div_(
                log_eigen@W@log_eigen + self.eps
            )

        else:
            self.weights.copy_(log_eigen)
            self.weights.div_(
                torch.sum(log_eigen**2) + self.eps
            )

    def update(self, alpha: Tensor):
        b = self.momentum
        structure = alpha.shape[:-2]
        alpha_r = alpha.view(-1, self.n_channels, self.n_dofs).contiguous()
        # [N, n_channel, n_dof]
        log_alpha = alpha_r.abs_().log10_()
        if len(structure) != 0:
            log_alpha = log_alpha.mean(dim=0) # [n_channel, n_dof]
        mean_log_alpha = log_alpha.mean(dim=-1, keepdim=True) # [n_channel, 1]
        self.s.lerp_((mean_log_alpha - log_alpha)@self.weights, 1-b)

    def forward(self, data: Tensor):
        assert data.dim() >= 2
        alpha = self.alpha(data)

        if self.training:
            self.update(alpha)

        V = self.V
        lam = self.w[None, :]
        slope = self.s[:, None]
        L = torch.pow(lam, slope)
        return torch.einsum('...cj, ij, cj -> ...ci', alpha, V, L)


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


### Double fractional modules ###

class SparkleFractional(_EigenvalueBase):
    def __init__(self, n_dofs: int, n_channels: int, *,
                 dtype: Optional[_dtype]=float64,
                 device: Union[_device, str, None]=None) -> None:
        super().__init__(n_dofs, dtype=dtype, device=device)
        kwargs = dict(dtype=dtype, device=device)
        self.s0 = Parameter(torch.empty((), **kwargs))
        self.s1 = Parameter(torch.empty((n_channels, ), **kwargs))
        self.reset_operator()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.s0, 0.)
        init.constant_(self.s1, 0.)

    @property
    def s(self) -> Tensor:
        return self.s0 + self.s1

    def update(self):
        with torch.no_grad():
            self.s0.add_(self.s1.mean())
            self.s1.sub_(self.s1.mean())

    def matrix(self):
        V = self.V
        Vinv = self.Vinv
        lam = self.w[None, :]
        slope = self.s[:, None]
        L = torch.pow(lam, slope)
        return torch.einsum('ij, cj, jk -> cik', V, L, Vinv)

    def forward(self, data: Tensor) -> Tensor: # [n_channel, n_dof] -> [n_channel, n_dof]
        if self.training:
            self.update()
        return torch.einsum('cik, ...ck -> ...ci', self.matrix(), data)
