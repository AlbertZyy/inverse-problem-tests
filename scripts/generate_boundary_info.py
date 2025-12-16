
from fealpy.backend import bm
from fealpy.mesh import TriangleMesh
from matplotlib import pyplot as plt

EXT = 63
H = 2./EXT

umesh = TriangleMesh.from_box([-1, 1, -1, 1], EXT, EXT)
NN = umesh.number_of_nodes()
node = umesh.entity('node')
isBDNode = umesh.boundary_node_flag()
N_BDRY_NODE = bm.sum(isBDNode)

# Edge
edge = umesh.entity('edge')
isBDEdge = umesh.boundary_edge_flag()

node_index_on_bdry = bm.zeros(NN, dtype=bm.int32)
node_index_on_bdry[isBDNode] = bm.arange(N_BDRY_NODE)

bdry_edge = node_index_on_bdry[edge[isBDEdge]]

edge_barycenter = umesh.entity_barycenter('edge', isBDEdge)
edge_normal = umesh.edge_unit_normal(isBDEdge)

import numpy as np

np.savez(
    'data/boundary_info_64x64.npz',
    node=node,
    is_bdry_node=isBDNode,
    bdry_edge=bdry_edge,
    bdry_edge_barycenter=edge_barycenter,
    bdry_edge_normal=edge_normal
)

# figure = plt.figure()
# axes = figure.add_subplot(111)
# umesh.add_plot(axes)
# umesh.find_node(axes, multiindex=node_index_on_bdry, showindex=True)

# plt.show()