
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle

# DATA_ID_LIST = np.random.randint(0, 2000, (20, )).tolist()
AXES_MAT = [3, 4]
HEAD = 0
fig = plt.figure(figsize=(12, 12))
x = np.linspace(-1, 1, 64)
y = np.linspace(-1, 1, 64)
X, Y = np.meshgrid(x, y, indexing='ij')

for pos, i in enumerate(range(HEAD, HEAD + math.prod(AXES_MAT))):
    axes: Axes = fig.add_subplot(*AXES_MAT, pos+1)
    file_ = np.load(f'lafem/data/cir3_e64_64_c8_fp32/inclusion/{i}.npz')
    ctrs, rads = file_['ctrs'], file_['rads']
    label = file_['label']
    if pos == HEAD:
        print("Label dtype: ", label.dtype)
    label = np.asarray(label, dtype=np.float64)
    if pos == HEAD:
        print("Label range: ", label.max(), label.min())
    axes.pcolormesh(X, Y, label.reshape(64, 64), cmap='jet', vmin=0., vmax=1.)

    for j in range(ctrs.shape[0]):
        circle = Circle((ctrs[j, 0], ctrs[j, 1]), rads[j], color='black', fill=False, linewidth=0.5)
        axes.add_patch(circle)

    axes.invert_yaxis()
    axes.set_xlim(-1, 1)
    axes.set_ylim(-1, 1)
    axes.set_title(i)
    # 关闭刻度显示
    axes.set_xticks([])
    axes.set_yticks([])

plt.tight_layout()
plt.show()
