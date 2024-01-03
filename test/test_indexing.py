
import sys

sys.path.append('./src')

import torch

from fdm import Indexing, UniformPartition, LaplaceFDMSolver
from dataset import NPZDataset

dataset = NPZDataset('./data/gdgn_64_64_train', 10)
gdgn, labels = dataset[0]
gd = gdgn[:, 0, :]
gn = gdgn[:, 1, :]
print(gd.shape)

EXT = 63
H = 2./EXT

solver = LaplaceFDMSolver([EXT, EXT], [H, H])
uh = solver.solve_from_gn(gn)
print(uh.shape)

un = solver.normal_derivative(uh)
print(un.shape)
print(torch.allclose(un, gn))

# from matplotlib import pyplot as plt

# fig = plt.figure()

# for i in range(8):
#     axes = fig.add_subplot(2, 4, i+1)
#     axes.imshow(uh[i, ...])

# plt.show()
