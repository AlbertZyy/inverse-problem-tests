
import sys

sys.path.append('./src')

import torch
from torch import Tensor
from fealpy.mesh import UniformMesh2d

from fdm import LaplaceFDMSolver


EXT = 63
H = 2./EXT

def solution(p: Tensor):
    x, y = p.split(1, dim=-1)
    # return x**2 - y**2
    return x*y

solver = LaplaceFDMSolver([EXT, EXT], [H, H])

mesh = UniformMesh2d([0, EXT, 0, EXT], [H, H], origin=[-1, -1])
bd_node = mesh.ds.boundary_node_flag()
node = torch.from_numpy(mesh.entity('node', index=bd_node))
gd = solution(node).flatten()

uh = solver.solve_from_gd(gd, reshape=True)


from matplotlib import pyplot as plt

fig = plt.figure()

axes = fig.add_subplot(111)
axes.imshow(uh.swapaxes_(0, 1), origin='lower')

plt.show()
