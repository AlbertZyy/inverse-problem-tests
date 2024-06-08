
import sys
sys.path.append("./src")

import numpy as np
import torch
from fealpy.torch.mesh import TriangleMesh

from fem import DataFeatureFEMSolver
from dataset import NPZDataset


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
EXT = 63

mesh = TriangleMesh.from_box([-1, 1, -1, 1], nx=EXT, ny=EXT, device=DEVICE)
df_solver = DataFeatureFEMSolver(mesh, p=1)
dataset = NPZDataset('D:\\Data\\gdgn_cir3_e64_64_c8_validate',
                     channel_keys=['gd'])
gn = np.load('D:\\Data\\gdgn_cir3_e64_64_c8_validate\\gn.npy')
gn = torch.from_numpy(gn)
gd = dataset[0][0]
data = torch.stack([gd, gn], dim=1).to(DEVICE)
output = df_solver(data.unsqueeze(0))
BATCH, CHANNEL = output.shape[:2]
output = output.reshape(BATCH, CHANNEL, EXT+1, EXT+1)


from matplotlib import pyplot as plt

fig = plt.figure(figsize=(12, 6))

for i in range(CHANNEL):
    axes = fig.add_subplot(2, 4, i+1)
    axes.pcolormesh(output[0, i, :, :].detach().cpu().numpy(), cmap='jet')

plt.show()
