
from fealpy.mesh import UniformMesh2d, IntervalMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import BilinearForm, ScalarMassIntegrator

EXTx = 4
EXTy = 4
Hx = 2./EXTx
Hy = 2./EXTy
Origin = [-1., -1.]
Q_ = 3

uniform_mesh = UniformMesh2d([0, EXTx, 0, EXTy], [Hx, Hy], origin=Origin)
mesh = IntervalMesh.from_mesh_boundary(uniform_mesh)
NN = mesh.number_of_nodes()
mesh.uniform_refine(1)
space = LagrangeFESpace(mesh, 1)

print(space.cell_to_dof())
bform_1 = BilinearForm(space)
bform_1.add_domain_integrator(ScalarMassIntegrator(q=Q_))
M = bform_1.assembly().toarray()
print(M[:NN, :NN])

from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
mesh.add_plot(ax)
mesh.find_node(ax, showindex=True)
mesh.find_cell(ax, showindex=True)
plt.show()
