
import sys

sys.path.append('./src')

import torch
from torch import Tensor
from fealpy.mesh import UniformMesh2d

from fdm import Indexing, UniformPartition, LaplaceFDMSolver
from dataset import NPZDataset

# dataset = NPZDataset('./data/gdgn_64_64_train', 10)
# gdgn, labels = dataset[0]
# gd = gdgn[1, 0, :]
# gn = gdgn[1, 1, :]
# print(gd.shape)

EXT = 63
H = 2./EXT

def solution(p: Tensor):
    x, y = p.split(1, dim=-1)
    return x**2 - y**2

solver = LaplaceFDMSolver([EXT, EXT], [H, H])

mesh = UniformMesh2d([0, EXT, 0, EXT], [H, H], origin=[-1, -1])
bd_node = mesh.ds.boundary_node_flag()
node = torch.from_numpy(mesh.entity('node', index=bd_node))
gd = solution(node).flatten()

uh = solver.solve_from_gd(gd, return_image=True)


from matplotlib import pyplot as plt

fig = plt.figure()

axes = fig.add_subplot(111, projection='3d')
mesh.show_function(axes, uh)

plt.show()
