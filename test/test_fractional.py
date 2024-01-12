
import sys
sys.path.append("./src")

import torch

from fractional import MultiChannelFractional, FractionalWithHighcut
from dataset import NPZDataset
from common import add_gaussian_noise

USE_NOISE = True

# frac = FractionalWithHighcut(252)
frac = MultiChannelFractional(252, 1, 1.25)
frac.from_npz(r"./data/laplace_beltrami_63_63.npz")
# frac.initialize(s=0.5, hc=391.72)

# 20% - 391.72
# 50% - 2448.27
frac.initialize(s=[0.8, ]*1, hc=[391.72, ]*1)

if USE_NOISE:
    gd = torch.randn(8, 252, dtype=torch.float64)
else:
    dataset = NPZDataset(r"./data/gdgn_64_64_train", 100)
    gd = dataset[51][0][7:8, 0, :]
    add_gaussian_noise(gd, 0.05)

alpha = frac.alpha(gd)
val = frac(gd)
alpha_2 = frac.alpha(val)


from matplotlib import pyplot as plt

fig = plt.figure()
axes = fig.add_subplot(1, 2, 1)
axes.plot(torch.log10(alpha.T.detach().abs()).numpy())
axes.set_ylim(-3., 3.)

axes = fig.add_subplot(1, 2, 2)
axes.plot(torch.log10(alpha_2.T.detach().abs()).numpy())
axes.set_ylim(-3, 3.)

plt.show()
