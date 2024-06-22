
import sys

sys.path.append('src')

import numpy as np
import torch
from torch.utils.data import RandomSampler, BatchSampler
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from fealpy.torch.mesh import TriangleMesh

from fem import EITDataPreprocessor, DataFeatureFEMSolver, LaplaceFEMSolver
from fdm import LaplaceFDMSolver
from data_feature import MultiChannelDataFeature
from dataset import NPYDataset, NPZDataset


fem_dataset = NPYDataset('data/cir3_e64_64_c8_test/gd', names=[str(i) for i in range(50)])
fem_gn = torch.from_numpy(np.load('data/cir3_e64_64_c8_test/gn.npy'))

fdm_dataset = NPZDataset('data/cir3_e64_64_c8_old_test',
                         names=[str(i) for i in range(50)],
                         channel_keys=['1', '2', '3', '4', '5', '6', '8', '16'])

INEDX = 9
CH = 0

mesh = TriangleMesh.from_box([-1, 1, -1, 1], nx=63, ny=63)
fem_solver = LaplaceFEMSolver(mesh, p=1, q=3)
fem_dfprepro = EITDataPreprocessor(fem_solver)
fem_dfsolver = DataFeatureFEMSolver(fem_solver)

fem_data = fem_dataset[INEDX]
fem_data = torch.stack([fem_data, fem_gn], dim=-2).unsqueeze(0)
fem_phi = fem_dfsolver(fem_dfprepro(fem_data)).reshape(8, 64, 64)

del mesh, fem_dfprepro, fem_dfsolver, fem_solver, fem_data

if not input('continue?') == 'y':
    exit(0)

fdm_solver = LaplaceFDMSolver([63, 63], [2/63, 2/63])
fdm_dfsolver = MultiChannelDataFeature(fdm_solver)

fdm_data, _ = fdm_dataset[INEDX]
fdm_data = fdm_data.unsqueeze(0)
fdm_phi = fdm_dfsolver(fdm_data).squeeze_(0)

del fdm_solver, fdm_dfsolver, fdm_data

if not input('continue?') == 'y':
    exit(0)

diff = (fem_phi - fdm_phi).abs_()
diff = diff.detach_().cpu().numpy()

### PLOT
x = np.linspace(-1, 1, 64)
y = np.linspace(-1, 1, 64)
X, Y = np.meshgrid(x, y, indexing='ij')

fig = plt.figure(figsize=(16, 4))

axes = fig.add_subplot(1, 3, 1)
qm = axes.pcolormesh(X, Y, diff[CH, :, :], cmap='inferno')
fig.colorbar(qm, ax=axes)
axes.set_title('diff.abs')

axes = fig.add_subplot(1, 3, 2)
qm = axes.pcolormesh(X, Y, fem_phi[CH, :, :], cmap='jet')
fig.colorbar(qm, ax=axes)
axes.set_title('fem')

file_ = np.load(f'data/cir3_e64_64_c8_test/inclusion/{INEDX}.npz')
ctrs, rads = file_['ctrs'], file_['rads']
for j in range(ctrs.shape[0]):
    circle = Circle((ctrs[j, 0], ctrs[j, 1]), rads[j], color='white', fill=False, linewidth=1.5, linestyle='--')
    axes.add_patch(circle)

axes = fig.add_subplot(1, 3, 3)
qm = axes.pcolormesh(X, Y, fdm_phi[CH, :, :], cmap='jet')
fig.colorbar(qm, ax=axes)
axes.set_title('fdm')

file_ = np.load(f'data/cir3_e64_64_c8_old_test/{INEDX}.npz')
ctrs, rads = file_['ctrs'], file_['rads']
for j in range(ctrs.shape[0]):
    circle = Circle((ctrs[j, 0], ctrs[j, 1]), rads[j], color='white', fill=False, linewidth=1.5, linestyle='--')
    axes.add_patch(circle)

plt.show()
