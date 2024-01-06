
import sys

sys.path.append('./src')

import torch
from torch import Tensor, cos

from fdm import LaplaceFDMSolver
from fractional import MultiChannelFractional
from data_feature import MultiChannelDataFeature
from dataset import NPZDataset

PI = torch.pi
EXT = 63
H = 2./EXT


def complex_color(complex_array: Tensor, base=1., allow_white=True):
    """
    (HSL mode) Set hue and lightness from angle and length of every complex number,
    while saturation is fixed to 240 (full).\n

    Returns
    ----
    `imgdata`: Tensor in a same shape by default.
        Every element stores RGB data, in [0, 1].
    """
    if not isinstance(complex_array, Tensor):
        raise TypeError(f'Tensor required, but got {type(complex_array)}.')

    angle = torch.angle(complex_array)
    l = abs(complex_array) / base
    if not allow_white:
        l = torch.tanh_(l)
    red = (cos(angle/2)**2 * (l+1-abs(l-1)) - 2) / (l+1+abs(l-1)) + 1
    green = (cos(angle/2-1/3*PI)**2 * (l+1-abs(l-1)) - 2) / (l+1+abs(l-1)) + 1
    blue = (cos(angle/2-2/3*PI)**2 * (l+1-abs(l-1)) - 2) / (l+1+abs(l-1)) + 1
    result = torch.stack((red, green, blue), dim=2)

    return result


def loged(x: Tensor):
    return 0.401*torch.log(11.107*x + 1)


solver = LaplaceFDMSolver([EXT, EXT], [H, H])
frac = MultiChannelFractional.from_npz("./data/laplace_beltrami_63_63.npz", n_channel=8)

data_feature = MultiChannelDataFeature(solver, frac)
dataset = NPZDataset("./data/gdgn_64_64_phrase2/", 10)
data, label = dataset[24]

val = data_feature(data[None, ...])

from matplotlib import pyplot as plt

fig = plt.figure()
freq = [2, 3, 5, 7]

for i in range(4):
    img1 = complex_color(val[0, 2*i, ...] + val[0, 2*i+1, ...]*1j, 1/freq[i])
    img1 = loged(img1)
    axes = fig.add_subplot(2, 2, i+1)
    axes.imshow(img1.detach().swapaxes_(0, 1), origin='lower')

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
axes.imshow(label.detach().swapaxes_(0, 1), origin='lower')

plt.show()
