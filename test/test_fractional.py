
import sys
sys.path.append("./src")

import torch

from fractional import MultiChannelFractional, Fractional
from dataset import NPZDataset
from common import add_gaussian_noise

# 2448.27
frac = MultiChannelFractional.from_npz(r"./data/laplace_beltrami_63_63.npz", n_channel=8, hc_slope=5.)
frac.set_initial_(slope=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], high_cut=[10000.0, ]*8)
frac.trainable_(False)

# dataset = NPZDataset(r"./data/gdgn_64_64_train", 100)
# gd = dataset[51][0][:, 0, :]
gd = torch.randn(8, 252, dtype=torch.float64)
# add_gaussian_noise(gd, 0., 0.1)
alpha = frac.alpha(gd)

val = frac(gd)
alpha_2 = frac.alpha(val)


from matplotlib import pyplot as plt

fig = plt.figure()
axes = fig.add_subplot(1, 2, 1)
axes.plot(alpha.T.detach().abs().numpy())

axes = fig.add_subplot(1, 2, 2)
axes.plot(alpha_2.T.detach().abs().numpy())

plt.show()
