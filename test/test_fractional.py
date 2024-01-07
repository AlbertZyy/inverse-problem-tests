
import sys
sys.path.append("./src")

import torch

from fractional import MultiChannelFractional, FractionalWithHighcut
from dataset import NPZDataset
from common import add_gaussian_noise

# frac = FractionalWithHighcut(252)
frac = MultiChannelFractional(252, 8)
frac.from_npz(r"./data/laplace_beltrami_63_63.npz")
# frac.initialize(s=0.5, hc=391.72)
frac.initialize(s=[0.5, ]*8, hc=[391.72, ]*8)

dataset = NPZDataset(r"./data/gdgn_64_64_train", 100)
gd = dataset[51][0][:, 0, :]
add_gaussian_noise(gd, 0.05)
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
