
import torch
from fealpy.mesh import UniformMesh2d


EXT = 63
I = torch.arange(4*EXT)
bd_index = torch.zeros_like(I)
bd_index[0     : EXT]   = torch.arange(0, EXT)
bd_index[EXT   : 2*EXT] = torch.arange(EXT, 3*EXT, 2)
bd_index[2*EXT : 3*EXT] = torch.arange(4*EXT-1, 3*EXT-1, -1)
bd_index[3*EXT : ]      = torch.arange(3*EXT-1, EXT, -2)


mesh = UniformMesh2d([0, 63, 0, 63], [1/63, 1/63])
bd_node = mesh.ds.boundary_node_index()
node = mesh.entity('node', bd_node)[bd_index]

from matplotlib import pyplot as plt

fig = plt.figure()
axes = fig.add_subplot(111)
mesh.add_plot(axes)
mesh.find_node(axes, node=node, showindex=True)

plt.show()
