
import os
import sys
sys.path.append('src')
from typing import Optional
from multiprocessing import Process, Queue

import numpy as np
import torch
from torch import Tensor
from fealpy.torch.mesh import TriangleMesh
from fealpy.torch import logger
from tqdm import tqdm

from fem import EITDataPreprocessor, LaplaceFEMSolver
from fractional import Fractional
from dataset import NPYDataset, DataLoader


logger.setLevel('WARNING')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
INPUT_FOLDER = 'data/cir3_e64_64_c8'
START = 0
END = 12000
OUTPUT_FOLDER = 'data/cir3_e64_64_c8/gnvn_g1'
NOISE = 0.01
use_noise_filter = False


if use_noise_filter:
    noise_filter = Fractional(252, device=DEVICE)
    noise_filter.from_npz(r"./data/laplace_beltrami_torch_63_63.npz")
    noise_filter.initialize(s=-0.75)
    noise_filter.s.requires_grad_(False)
else:
    noise_filter = None


def process_data(queue):
    dataset = NPYDataset(os.path.join(INPUT_FOLDER, 'gd'),
                         names=[f"{i}" for i in range(START, END)])
    loader = DataLoader(dataset, batch_size=100, drop_last=False)
    gn = torch.from_numpy(
        np.load(os.path.join(INPUT_FOLDER, 'gn.npy'))
    ).to(DEVICE).unsqueeze(0)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    mesh = TriangleMesh.from_box([-1, 1, -1, 1], nx=63, ny=63, device=DEVICE)
    solver = LaplaceFEMSolver(mesh, p=1)
    processor = EITDataPreprocessor(solver)

    for gd in tqdm(loader):
        gd = gd.to(DEVICE)
        noise = torch.randn_like(gd) * NOISE
        if noise_filter:
            noise = noise_filter(noise)
        gd += gd * noise
        gn_ = gn.broadcast_to(gd.shape)
        data = torch.stack([gd, gn_], dim=-2) # [B, CH, 2, bddof]
        gnvn = processor(data).cpu().numpy()
        queue.put(gnvn)

    queue.put(None)


def main():
    q: Queue[Optional[Tensor]] = Queue()
    p = Process(target=process_data, args=(q,))
    p.start()
    index = 0

    while True:
        item = q.get()

        if item is None:
            break

        for i in range(item.shape[0]):
            np.save(os.path.join(OUTPUT_FOLDER, f'{index}.npy'), item[i, ...])
            index += 1


if __name__ == '__main__':
    main()
