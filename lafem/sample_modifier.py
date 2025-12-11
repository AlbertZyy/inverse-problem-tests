
import numpy as np
from numpy.typing import NDArray as Tensor
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import os


FREQ = [1, 2, 3, 4, 5, 6, 8, 16]

def levelset(p: Tensor, centers: Tensor, radius: Tensor):
    """Calculate level set function value."""
    struct = p.shape[:-1]
    p = p.reshape(-1, p.shape[-1])
    dis = np.linalg.norm(p[:, None, :] - centers[None, :, :], axis=-1) # (N, NCir)
    ret = np.min(dis - radius[None, :], axis=-1) # (N, )
    return ret.reshape(struct)


def neumann(points: Tensor, *args):
    x = points[..., 0]
    y = points[..., 1]
    kwargs = {'dtype': points.dtype}
    theta = np.arctan2(y, x)
    freq = np.array(FREQ, **kwargs)
    return np.sin(np.tensordot(freq, theta, axes=0))


def show_inclusion(axes, label, ctrs, rads):
    x = np.linspace(-1, 1, 64)
    y = np.linspace(-1, 1, 64)
    X, Y = np.meshgrid(x, y, indexing='ij')

    axes.pcolormesh(X, Y, label.reshape(64, 64), cmap='jet', vmin=0, vmax=1)

    for i in range(len(rads)):
        axes.text(ctrs[i, 0], ctrs[i, 1], f'{i}', color='white', fontsize=16)
        axes.add_patch(Circle(ctrs[i], rads[i], color='white', fill=False, linewidth=1.25, linestyle='--'))

    axes.set_aspect('equal')
    axes.set_title('label')


PATH = "lafem/data/cir5_e64_64_c8"
ID = 105


sample_path = os.path.join(PATH, f"inclusion/{ID}.npz")
inclusion_data = np.load(sample_path)
label = inclusion_data['label']
rads = inclusion_data['rads']
ctrs = inclusion_data['ctrs']

fig = plt.figure()
axes = fig.add_subplot(121)
show_inclusion(axes, label, ctrs, rads)

axes = fig.add_subplot(122)
new_label = label.copy()
new_ctrs = ctrs.copy()
new_rads = rads.copy()
new_ctrs[3, 1] = -0.303286853779
show_inclusion(axes, new_label, new_ctrs, new_rads)

plt.show()

signal = input("Continue? (y/n)")

if signal != 'y':
    exit()


from fealpy.mesh import TriangleMesh, UniformMesh2d
from fealpy.cem.generator import EITDataGenerator

EXT = 63
H = 2./EXT
P = 1
Q = 4
SIGMA = [10., 1.]
OUTPUT_PATH = "lafem/data/cir5_e64_64_c8_modified"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'gd'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'inclusion'), exist_ok=True)

umesh = UniformMesh2d([0, EXT, 0, EXT], [H, H], [-1., -1.], itype=np.int32, ftype=np.float64)
ls_fn = lambda p: levelset(p, new_ctrs, new_rads)
interface_mesh = TriangleMesh.interfacemesh_generator(umesh, phi=ls_fn)
generator = EITDataGenerator(mesh=interface_mesh, p=P, q=Q)
gn = generator.set_boundary(neumann, batch_size=len(FREQ))
pixel = umesh.entity('node')
label = generator.set_levelset(SIGMA, ls_fn, pixel)
gd = generator.run()

np.save(
    os.path.join(OUTPUT_PATH, f'gd/{ID}.npy'),
    gd
)
np.savez(
    os.path.join(OUTPUT_PATH, f'inclusion/{ID}.npz'),
    label=label,
    ctrs=new_ctrs,
    rads=new_rads
)
