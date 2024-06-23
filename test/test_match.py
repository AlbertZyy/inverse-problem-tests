
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

INEDX = 33
CH = 0

mesh = TriangleMesh.from_box([-1, 1, -1, 1], nx=63, ny=63)
fem_solver = LaplaceFEMSolver(mesh, p=1, q=4)
fem_dfprepro = EITDataPreprocessor(fem_solver)
fem_dfsolver = DataFeatureFEMSolver(fem_solver)

# fem_data = fem_dataset[INEDX]
# fem_data = torch.stack([fem_data, fem_gn], dim=-2).unsqueeze(0)
# NOTE: 两个数据集的 gn 基本就相差一个网格的 h。
# 用以下的 fdm 数据 *h 测试 fem 模型，能得出 fem 数据下几乎一样的结果。
fem_data, _ = fdm_dataset[INEDX]
fem_data = fem_data.unsqueeze(0)
fem_data[:, :, 1, :] = fem_data[:, :, 1, :] * 2./63

fem_phi = fem_dfprepro(fem_data)
fem_phi = fem_dfsolver(fem_phi)
fem_phi = fem_phi.reshape(8, 64, 64)

# del mesh, fem_dfprepro, fem_dfsolver, fem_solver, fem_data

if not input('continue?') == 'y':
    exit(0)

fdm_solver = LaplaceFDMSolver([63, 63], [2/63, 2/63])
fdm_dfsolver = MultiChannelDataFeature(fdm_solver)

fdm_data, _ = fdm_dataset[INEDX]
fdm_data = fdm_data.unsqueeze(0)
fdm_phi = fdm_dfsolver(fdm_data).squeeze_(0)
fdm_phi = fdm_phi.reshape(8, 64, 64)

# del fdm_solver, fdm_dfsolver, fdm_data

if not input('continue?') == 'y':
    exit(0)

# NOTE: 如果 phi 的边界条件仅包含 gn，则差异在 1e-5 量级。
# 基本可以判定误差产生于 $\nabla v \cdot \bm{n}$

# NOTE: 比较由 gd 求出的 v：当使用各自的数据集时，差异在 1e-4 量级；
# 当都使用 fdm 数据集里的 gd 时，差异在 1e-14 量级。
# *这也是个问题，因为理论上两个数据集的 gd 应当是一样的。

# NOTE: 因此可以确定，是两种法向导数的计算存在较大的差异。哪种有问题？
# 当用 vn 作为 neumann bc 计算 phi 时，理论上 phi 就等于 v。
# 检查两种方法的 v 是否接近这中特殊的 phi。
# FDM 模型的差异 1e-14；FEM 模型的差异 1e-2。
# 因此我们可以确定，是 FEM 模型的法向导数计算存在问题。

diff = (fem_phi - fdm_phi)
diff = diff.detach_().cpu().numpy()

### PLOT
x = np.linspace(-1, 1, 64)
y = np.linspace(-1, 1, 64)
X, Y = np.meshgrid(x, y, indexing='ij')

fig = plt.figure(figsize=(16, 4))

# AXES 1
axes = fig.add_subplot(1, 3, 1)
qm = axes.pcolormesh(X, Y, diff[CH, :, :], cmap='inferno')
fig.colorbar(qm, ax=axes)
axes.set_title('diff.abs')

# AXES 2 FEM
axes = fig.add_subplot(1, 3, 2)
qm = axes.pcolormesh(X, Y, fem_phi[CH, :, :], cmap='jet')
fig.colorbar(qm, ax=axes)
axes.set_title('fem')

file_ = np.load(f'data/cir3_e64_64_c8_test/inclusion/{INEDX}.npz')
ctrs, rads = file_['ctrs'], file_['rads']
for j in range(ctrs.shape[0]):
    circle = Circle((ctrs[j, 0], ctrs[j, 1]), rads[j], color='white', fill=False, linewidth=1.5, linestyle='--')
    axes.add_patch(circle)

# AXES 3 FDM
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
