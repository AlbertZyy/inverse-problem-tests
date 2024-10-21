
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

# DATA_ID_LIST = np.random.randint(0, 2000, (20, )).tolist()
AXES_MAT = [8, 10]
fig = plt.figure(figsize=(12, 12))

for pos, i in enumerate(range(2000, 2080)):
    axes = fig.add_subplot(*AXES_MAT, pos+1)
    file_ = np.load(f'lafem/data/cir3_e64_64_c8/inclusion/{i}.npz')
    ctrs, rads = file_['ctrs'], file_['rads']

    for j in range(ctrs.shape[0]):
        circle = Circle((ctrs[j, 0], ctrs[j, 1]), rads[j], color='black', fill=False, linewidth=1.0)
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
