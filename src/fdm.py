
from functools import reduce
from typing import Sequence, Optional, Callable, Union, Literal, TypeVar

import torch
from torch import Tensor, int32, float64, device
from torch.linalg import lu_factor, lu_solve


IndexFunc = Callable[[Tensor], Tensor]
IndexFuncOrTensor = Union[Tensor, IndexFunc]
BlockIndex = Sequence[Literal[-1, 0, 1]]

_Self = TypeVar('_Self')
_RT = TypeVar('_RT')


def enable_cache(meth: Callable[[_Self], _RT]):
    def wrapper(self: _Self, *, refresh=False) -> _RT:
        name = '_cache_' + meth.__name__

        if (not refresh) and hasattr(self, name):
            return getattr(self, name)
        else:
            return_value = meth(self)
            setattr(self, name, return_value)
            return return_value

    return wrapper


class Indexing():
    def __init__(self, steps: Sequence[int],
                 *, itype=int32, device: device=None) -> None:
        """
        @brief Build a indexing system in a N-d uniform mesh.

        @param steps: Sequence[int]. Number of nodes along each dimension.
        @param margin: int. Distance from the working area to the real boundary.
        """
        GD = len(steps)
        _NN = reduce(lambda x, y:x*y, steps)
        _node = torch.arange(_NN, dtype=itype, device=device).reshape(steps)
        self.GD = GD
        self.NN = _NN
        self.node = _node
        self.steps = steps
        self.device = device

    @property
    def shape(self):
        return self.node.shape

    @property
    def itype(self):
        return self.node.dtype

    def shifted(self, dim: int, movement: int):
        """
        @brief Return the index of nodes in the area shifted towards a specific\
               direction.
        """
        index = [slice(None, None, None), ] * self.GD
        if movement >= 0:
            index[dim] = slice(movement, None, None)
        else:
            index[dim] = slice(None, movement, None)
        return self.node[index].flatten()

    def layer(self, dim: int, layer: int):
        """
        @brief Return the index of nodes on a specific layer perpendicular to a\
               specific direction.
        """
        index = [slice(None, None, None), ] * self.GD
        index[dim] = layer
        return self.node[index].flatten()

    @enable_cache
    def boundary_flag(self):
        """
        @brief Return a bool Tensor indicating if nodes are on the boundary.
        *frequently used*, *cache*.
        """
        flag = torch.zeros((self.NN, ), dtype=torch.bool, device=self.device)
        for d in range(self.GD):
            flag[self.layer(d, 0)] = True
            flag[self.layer(d, -1)] = True
        return flag

    def boundary(self):
        """
        @brief Return the index of nodes on the boundary.
        """
        return torch.nonzero(self.boundary_flag())[..., 0]

    def interior(self, padding: int=0):
        """
        @brief Return the index of all interior nodes.
        """
        if padding == 0:
            index = [slice(None, None, None), ] * self.GD
        elif padding >= 1:
            index = [slice(padding, -padding, None)] * self.GD
        else:
            raise ValueError("padding must can not be negative.")
        return self.node[index].flatten()

    def shift(self, index: Tensor, dim: int, step: int, *, inplace=False):
        """
        @brief Moves the given index array in a specific direction in the grid.
        """
        assert dim < self.GD and dim >= -self.GD
        if not inplace:
            index = index.clone()

        index += self.node.stride(dim) * step
        return index

    def index_grid(self):
        """
        @brief Return the mesh grid of the node positions.
        """
        mesh = torch.meshgrid(
            [torch.arange(0, e, device=self.device) for e in self.steps],
            indexing='ij'
        )
        return mesh


class UniformPartition(Indexing):
    @classmethod
    def from_uniform_mesh(cls, mesh):
        extent = mesh.extent
        GD = len(extent) // 2
        size = [extent[2*i+1] - extent[2*i] for i in range(GD)]
        h = mesh.h
        # NOTE: for UniformMesh1d
        if isinstance(h, (float, int)):
            h = [h, ]

        return cls(size=size, h=h)

    def __init__(self, size: Sequence[int], h: Sequence[float], *, itype=int32,
                 dtype=float64, device: device = None) -> None:
        """
        @brief Build a partition system in a N-d uniform mesh.

        @param size: Sequence[int]. Number of cells along each dimension.
        @param h: Sequence[float]. Length of cells along each dimension.
        """
        steps = [s + 1 for s in size]
        super().__init__(steps, itype=itype, device=device)
        self.h = torch.tensor(h, dtype=dtype, device=device)

    @property
    def dtype(self):
        return self.h.dtype

    def diffusion(self):
        """
        @brief Assemble diffusion operator matrix.
        """
        h = self.h
        cdim = 1. / h**2
        diag = (2 * cdim.sum()).broadcast_to(self.NN)
        A = torch.diagflat(diag)

        for d in range(self.GD):
            I = self.shifted(d, -1) # lower
            J = self.shifted(d, 1) # higher
            A[I, J] -= cdim[d]
            A[J, I] -= cdim[d]

        return A

    def dirichlet(self, A: Tensor):
        """
        @brief An in-place operation to apply dirichlet bc to matrix A.
        """
        I = self.boundary()
        A[I, :] = 0.
        A[I, I] = 1.
        return A

    @enable_cache
    def diffusion_dirichlet(self):
        """
        @brief Assemble diffusion operator matrix with dirichlet condition.
        """
        return self.dirichlet(self.diffusion())

    def dirichlet_source(self, gd: Tensor, *, out: Optional[Tensor]=None):
        """
        *frequently used*.
        """
        if out is None:
            dtype = self.dtype
            device = self.device
            b = torch.zeros((self.NN, ), dtype=dtype, device=device)
        else:
            assert out.ndim == 1
            b = out

        bd_node = self.boundary()
        return b.index_add(dim=-1, index=bd_node, source=gd)

    def neumann(self, A: Tensor):
        h = self.h
        cdim = 1. / h**2

        for d in range(self.GD):
            I = self.layer(d, 0)
            J = self.layer(d, 1)
            A[I, J] -= cdim[d]
            I = self.layer(d, -1)
            J = self.layer(d, -2)
            A[I, J] -= cdim[d]

        return A

    @enable_cache
    def diffusion_neumann(self):
        """
        @brief Assemble diffusion operator matrix with neumann condition.
        """
        return self.neumann(self.diffusion())

    def neumann_source(self, gn: Tensor, *, out: Optional[Tensor]=None):
        """
        *frequently used*.
        """
        if out is None:
            dtype = self.dtype
            device = self.device
            b = torch.zeros((self.NN, ), dtype=dtype, device=device)
        else:
            assert out.ndim == 1
            b = out

        bd_node = self.boundary()
        gn = gn * self._neumann_source_scale()
        return b.index_add(dim=-1, index=bd_node, source=gn)

    @enable_cache
    def _neumann_source_scale(self):
        h = self.h
        dtype = self.dtype
        device = self.device
        cdim = 2. / h
        b = torch.zeros((self.NN, ), dtype=dtype, device=device)
        for d in range(self.GD):
            I = self.layer(d, 0)
            b[I] += cdim[d]
            I = self.layer(d, -1)
            b[I] += cdim[d]
        return b[self.boundary_flag()]

    def coordinate(self, origin: Sequence[float]):
        """
        @brief Construct coordinate Tensor in shape (GD, Nx[, Ny[, Nz...]]).
        """
        GD = self.GD
        assert GD == len(origin)
        dtype = self.dtype
        device = self.device
        if isinstance(origin, Tensor):
            ori = origin
        else:
            ori = torch.tensor(origin, dtype=dtype, device=device)
        mesh = torch.stack(self.index_grid(), dim=0)
        coor = torch.einsum("d..., d -> d...", mesh, self.h)
        coor += ori.reshape((GD, ) + (1, )*GD)
        return coor


class LaplaceFDMSolver():
    """A finite difference method solver on a N-d box for the Laplace\
       equation $-\\Delta u = 0$."""
    def __init__(self, size: Sequence[int], h: Sequence[float], *, itype=int32,
                 dtype=float64, device: device=None) -> None:
        """
        @brief Build the FDM solver on a given N-d box.

        @param size: Sequence[int].
        @param h: Sequence[float].
        """
        self.dtype = dtype
        self.device = device
        self.indexing = UniformPartition(size, h, itype=itype, dtype=dtype, device=device)

    def _init_gd(self):
        indexing = self.indexing
        A_d = indexing.diffusion_dirichlet()
        self.A_d_LU, self.pivots_d = lu_factor(A_d)

    def solve_from_gd(self, gd: Tensor, *, flatten=True) -> Tensor:
        """
        @brief Solve the Laplace equation from a dirichlet boundary data.
        """
        if not hasattr(self, 'A_d_LU'):
            self._init_gd()

        A_d_LU = self.A_d_LU
        pivots_d = self.pivots_d
        b_ = self.indexing.dirichlet_source(gd)[:, None]

        uh = lu_solve(A_d_LU, pivots_d, b_)
        if flatten:
            return uh[:, 0]
        else:
            return uh.reshape(self.indexing.shape)

    def _init_gn(self):
        indexing = self.indexing
        NN = indexing.NN
        A_n_c = torch.empty((NN+1, NN+1), dtype=self.dtype, device=self.device)
        A_n = indexing.diffusion_neumann()
        c = indexing.boundary_flag().to(dtype=torch.int)
        A_n_c[:-1, :-1] = A_n
        A_n_c[-1, :-1] = c
        A_n_c[:-1, -1] = c
        A_n_c[-1, -1] = 0.
        self.A_n_c = A_n_c
        self.A_n_LU, self.pivots_n = lu_factor(A_n_c)

    def solve_from_gn(self, gn: Tensor, *, flatten=True) -> Tensor:
        """
        @brief Solve the Laplace equation from a neumann boundary data.
        """
        if not hasattr(self, 'A_n_LU'):
            self._init_gn()

        A_n_LU = self.A_n_LU
        pivots_n = self.pivots_n
        b__ = self.indexing.neumann_source(gn)
        ZERO_ = torch.zeros((1, ), dtype=self.dtype, device=self.device)
        b_ = torch.cat([b__, ZERO_], dim=0)
        b_.unsqueeze_(-1)

        uh = lu_solve(A_n_LU, pivots_n, b_)[:-1]
        if flatten:
            return uh[:, 0]
        else:
            return uh.reshape(self.indexing.shape)

    def normal_derivative(self, uh: Tensor):
        """
        @brief Calculates the directional derivative of the function along the\
               outer normal direction on the boundary.
        """
        A = self.indexing.diffusion_neumann()
        val = A @ uh.flatten()
        is_bd_node = self.indexing.boundary_flag()
        val[is_bd_node] /= self.indexing._neumann_source_scale()
        return val[is_bd_node]

    # def Solve_from_GD(self):
    #     """Construct a PyTorch Module to solve from gD."""
    #     return _Modulized_GD(self)

    # def Solve_from_GN(self):
    #     """Construct a PyTorch Module to solve from gN."""
    #     return _Modulized_GN(self)

    # def Normal_Derivative(self):
    #     """Construct a PyTorch Module to calculate the directional derivative\
    #        along the normal direction of the boundary."""
    #     return _Modulized_Normal(self)

    # def Restore_Shape(self):
    #     """Construct a PyTorch Module to reshape the solution uh to the node."""
    #     return _Reshaper(self.indexing.node.shape)


# from torch.nn import Module
# class _Modulized(Module):
#     def __init__(self, caller: LaplaceFDMSolver) -> None:
#         super().__init__()
#         self.caller = caller
# class _Modulized_GD(_Modulized):
#     def forward(self, data: Tensor):
#         return self.caller.solve_from_gd(data)
# class _Modulized_GN(_Modulized):
#     def forward(self, data: Tensor):
#         return self.caller.solve_from_gn(data)
# class _Modulized_Normal(_Modulized):
#     def forward(self, data: Tensor):
#         return self.caller.normal_derivative(data)
# class _Reshaper(Module):
#     def __init__(self, size: Sequence[int]) -> None:
#         super().__init__()
#         self.size = size
#     def forward(self, data: Tensor):
#         return data.reshape(self.size)

if __name__ == "__main__":
    indexing = UniformPartition([10, ], [0.1, ])
    print(indexing.coordinate([0., ]))
