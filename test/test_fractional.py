
import sys
sys.path.append("./src")

import torch
import numpy as np

from fractional import MultiChannelFractional, Fractional


frac = MultiChannelFractional.from_npz(r"./data/laplace_beltrami_63_63.npz", n_channel=8)
frac.set_initial_(slope=[0.25, ]*8, high_cut=[500., ]*8)
frac.trainable_(False)
print(frac.Vinv@frac.V)

data = torch.randn(8, 252, dtype=torch.float64)
alpha = frac.alpha(data)
val = frac(data)
alpha_2 = frac.alpha(val)


from matplotlib import pyplot as plt

fig = plt.figure()
axes = fig.add_subplot(1, 2, 1)
axes.plot(alpha.T.detach().abs().numpy())

axes = fig.add_subplot(1, 2, 2)
axes.plot(alpha_2.T.detach().abs().numpy())

plt.show()
