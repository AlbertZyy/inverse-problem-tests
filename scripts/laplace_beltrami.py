"""
此脚本用于生成求解区域的边界 LB 算子的特征值和特征向量，并存储为单个 .npz 文件。
"""

from typing import Tuple
import argparse

import yaml
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import (
    BilinearForm,
    ScalarDiffusionIntegrator,
    ScalarMassIntegrator
)
from fealpy.mesh import IntervalMesh
from fealpy.mesh import UniformMesh2d


parser = argparse.ArgumentParser()
parser.add_argument("config", type=str, help="config file")
parser.add_argument("--plot", help="plot the eigenvalues", action="store_true")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

Q_ = config['fem']['integral']
assert isinstance(Q_, int)
assert Q_ >= 1

def laplace_eigen_fem(mesh) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    @brief
    """
    space = LagrangeFESpace(mesh, p=1)
    bform_0 = BilinearForm(space)
    bform_0.add_domain_integrator(ScalarDiffusionIntegrator(q=Q_))
    A = bform_0.assembly().toarray()
    bform_1 = BilinearForm(space)
    bform_1.add_domain_integrator(ScalarMassIntegrator(q=Q_))
    M = bform_1.assembly().toarray()
    w, v = eigh(A, M)
    return w, v, A, M

EXTx, EXTy = config['mesh']["ext"]
Lx, Ly = config['mesh']['length']
Hx = Lx/EXTx
Hy = Ly/EXTy
Origin = config['mesh']['origin']
N_REFINE = config['mesh']['refine']


print("Generating Boundary Laplace Beltrami operator...")
print(f"Config:")
print(f"  - Domain: [{Origin[0]}, {Origin[0] + Lx}]x[{Origin[1]}, {Origin[1] + Ly}]")
print(f"  - Mesh: {EXTx}x{EXTy}, refinement: {N_REFINE}")
print(f"  - Integral points: {Q_}")
print(f"will be saved to file: {config['file']}", end='\n\n')
signal_ = input("Continue? (y/n) ")


if signal_ in {'y', 'Y'}:

    uniform_mesh = UniformMesh2d([0, EXTx, 0, EXTy], [Hx, Hy], origin=Origin)
    mesh = IntervalMesh.from_mesh_boundary(uniform_mesh)
    del uniform_mesh

    NN = mesh.number_of_nodes()
    mesh.uniform_refine(N_REFINE)

    w, v, _, M = laplace_eigen_fem(mesh)

    w = w[1:NN+1]
    vinv = (v.T @ M)[1:NN+1, :NN] * N_REFINE**2
    v = v[:NN, 1:NN+1]

    np.savez(config['file'], w=w, v=v, vinv=vinv)
    print("Saved.")

    if args.plot:
        from matplotlib import pyplot as plt

        fig = plt.figure()

        axes = fig.add_subplot(121)
        f = np.arange(0, NN)
        axes.plot(f, np.sqrt(w))
        axes.plot(f, (f+1)/2*np.pi/4)

        axes = fig.add_subplot(122)
        mesh.add_plot(axes)
        mesh.find_node(axes, index=slice(None, NN), showindex=True)
        plt.show()

else:
    print("Aborted.")
