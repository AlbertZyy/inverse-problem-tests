
import sys
sys.path.append("./src")

import torch
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from fractional import AdaptiveFractional, Fractional
from data_feature import MultiChannelDataFeature
from dataset import NPZDataset

NOISE = 0.42
EXT = 63
H = 2./EXT

noise_filter = Fractional(252)
noise_filter.from_npz(r"./data/laplace_beltrami_63_63.npz")
noise_filter.initialize(s=-0.75)
noise_filter.s.requires_grad_(False)
frac = AdaptiveFractional(252, 8, momentum=0.9)
frac.from_npz(r"./data/laplace_beltrami_63_63.npz")
df = MultiChannelDataFeature.from_domain([EXT, EXT], [H, H], frac)

dataset = NPZDataset(r"./data/gdgn_64_64_train", 1000)
loader = DataLoader(dataset, batch_size=100, shuffle=True)

s_list = []

for epoch in trange(4):
    for data, _ in tqdm(loader, desc=f"Epoch {epoch+1}", leave=False):
        noise = torch.randn_like(data[:, :, 0, :]) * NOISE
        noise = noise_filter(noise)
        noise = data[:, :, 0, :] * noise
        data[:, :, 0, :] += noise
        gnvn = df.gd2gn_diff(data)
        gnvn = frac(gnvn)
        s_list.append(frac.s.detach().clone())

sdata = torch.stack(s_list, dim=0)

from matplotlib import pyplot as plt

fig = plt.figure()
axes = fig.add_subplot(111)
axes.plot(sdata.detach())
plt.show()

# PI = torch.pi

# length = alpha_1.shape[1] // 2
# x = torch.arange(1, length+1, dtype=torch.float32) * PI/4

# fig = plt.figure('data_feature_bd_spectrum', figsize=[18, 6])
# fig.suptitle("Energy spectrum of boundary value "
#              f"(noise={NOISE:.2e})")

# axes = fig.add_subplot(1, 3, 1)
# axes.plot(x, sum_energy(alpha_0).T.detach())
# axes.grid(True)
# axes.set_xlabel('Frequency')
# axes.set_xscale('log')
# axes.set_ylim(1e-6, 1e+2)
# axes.set_ylabel('Energy')
# axes.set_yscale('log')
# axes.set_title('Spectrum of gd')
# axes.legend([1, 2, 3, 4, 5, 6, 8, 16])

# axes = fig.add_subplot(1, 3, 2)
# axes.plot(x, sum_energy(alpha_1).T.detach())
# axes.grid(True)
# axes.set_xlabel('Frequency')
# axes.set_xscale('log')
# axes.set_ylim(1e-6, 1e+2)
# axes.set_ylabel('Energy')
# axes.set_yscale('log')
# axes.set_title('Spectrum of gn-vn')

# axes = fig.add_subplot(1, 3, 3)
# axes.plot(x, sum_energy(alpha_2).T.detach())
# axes.grid(True)
# axes.set_xlabel('Frequency')
# axes.set_xscale('log')
# axes.set_ylim(1e-6, 1e+2)
# axes.set_ylabel('Energy')
# axes.set_yscale('log')
# axes.set_title('Spectrum of after-frac gn-vn')

# plt.show()
