
import numpy as np
from matplotlib import pyplot as plt

# type_ = "Multi"
# tag = "nn_multi"
# gamma1 = np.linspace(0., 1., 20)
# gamma2 = np.linspace(0., 1., 20)
# G1, G2 = np.meshgrid(gamma1, gamma2, indexing='ij')
# result = np.load(f'lafem/landscape_1_2_{tag}.npy')

# fig = plt.figure(figsize=(6, 6))
# axes = fig.add_subplot(111)
# cm = axes.pcolormesh(G1, G2, result, cmap='jet')

# axes.set_xlabel(r'$\gamma_0$')
# axes.set_ylabel(r'$\gamma_1$')
# axes.set_title(f'Landscape of cross-entropy loss for {type_}')
# fig.colorbar(cm)
# plt.tight_layout()
# plt.savefig(f'lafem/landscape_1_2_{tag}.png')
# plt.show()


type_ = "Single"
tag = "nn_single"
gamma1 = np.linspace(0., 1., 20)
result = np.load(f'lafem/landscape_1_{tag}.npy')

fig = plt.figure(figsize=(6, 6))
axes = fig.add_subplot(111)
axes.plot(gamma1, result, label=r'$\gamma$')

axes.set_xlabel(r'$\gamma$')
axes.set_ylabel(r'Cross Entropy Loss')
axes.set_title(f'Landscape of cross-entropy loss for {type_}')
plt.tight_layout()
plt.savefig(f'lafem/landscape_1_{tag}.png')
plt.show()
