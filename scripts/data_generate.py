"""
此脚本用于生成数据集，每个样本包含多个 gD & gN 数据通道和标签。
求解区域为 [-1, 1], [-1, 1]。
"""

from typing import Sequence
from time import time
import argparse

import numpy as np
from numpy.random import rand
from numpy.typing import NDArray
from fealpy.cem.generator import LaplaceDataGenerator2d
import yaml
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("config", help="path of the .yaml file")


def levelset(p: NDArray, centers: NDArray, radius: NDArray):
    """
    Calculate level set function value.
    """
    struct = p.shape[:-1]
    p = p.reshape(-1, p.shape[-1])
    dis = np.linalg.norm(p[:, None, :] - centers[None, :, :], axis=-1) # (N, NCir)
    ret = np.min(dis - radius[None, :], axis=-1) # (N, )
    return ret.reshape(struct)


args = parser.parse_args()
with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


EXT = config['data']['ext']
SIGMA = config['data']['sigma']
FREQ = config['data']['freq']
PHR = config['data']['phrase']
OMEGA_NAME = config['data']['ch_names']
assert len(OMEGA_NAME) == len(FREQ) * len(PHR)
DTYPE = config['data']['dtype']
output_folder = config['output_folder']

if output_folder[-1] != "/":
    output_folder += "/"

def main(sigma_iterable: Sequence[int], seed=0, index=0):
    np.random.seed(seed)

    for sigma_idx in tqdm(sigma_iterable,
                          desc=f"worker{index}",
                          unit='sample',
                          position=index):

        count = 0

        while count < 100:

            ctrs = rand(3, 2) * 1.6 - 0.8 # (NCir, GD)
            b = np.min(0.9-np.abs(ctrs), axis=-1) # (NCir, )
            rads = rand(3) * (b-0.1) + 0.1 # (NCir, )
            ls_fn = lambda p: levelset(p, ctrs, rads)

            generator = LaplaceDataGenerator2d.from_cos(
                box=[-1, 1, -1, 1],
                nx=EXT, ny=EXT,
                levelset=ls_fn,
                sigma_vals=SIGMA,
                freq=FREQ,
                phrase=PHR
            )
            count += 1

            if generator.is_available():
                count = 100
                generator.save(sigma_idx,
                            output_folder,
                            OMEGA_NAME,
                            dtype=np.dtype(DTYPE))


def estimate_space_occupied(n_float: int, n_bool: int, n_samples: int, dtype: str):
    if dtype == "float32":
        single = n_float * 4 + n_bool
    elif dtype == "float64":
        single = n_float * 8 + n_bool
    else:
        raise ValueError(f"Unsupported dtype '{DTYPE}'.")
    return n_samples * single


def _unit(bytes: int):
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(np.floor(np.log(bytes) / np.log(1024)))
    if i == 0:
        return f"{bytes:.2f} {size_name[i]}"
    else:
        return f"{bytes / 1024 ** i:.2f} {size_name[i]}"


if __name__ == "__main__":
    import os

    process_num = config['process_num']
    space_occ = estimate_space_occupied(EXT * 8 * len(OMEGA_NAME),
                                        (EXT+1)**2,
                                        config['tail'] - config['head'],
                                        DTYPE)

    print("Start generating data...")
    print(f"using {process_num} processes")
    print("Config:")
    print(f"    mesh: {EXT}x{EXT}")
    print(f"    sigma(inclusion): {SIGMA[0]}")
    print(f"    sigma(background): {SIGMA[1]}")
    print(f"    freq: {FREQ}")
    print(f"    phrase: {PHR}")
    print(f"    dtype: {DTYPE}")
    print("Output:")
    print(f"    data shape: {2}x{EXT*4}")
    print(f"    n_channel: {len(OMEGA_NAME)}")
    print(f"    label shape: {EXT+1}x{EXT+1}")
    print(f"    data index range: {config['head']}~{config['tail']}")
    print(f"Space occupation estimate: {_unit(space_occ)},")
    print(f"will be saved to folder: {output_folder}", end='\n\n')
    signal_ = input("Continue? (y/n)")

    if signal_ not in {'y', 'Y'}:
        print("Aborted.")
        exit(0)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    from multiprocessing import Pool
    from fealpy.ml import timer

    pool = Pool(process_num)
    tmr = timer()
    tmr.send(None)

    NUM = tuple(range(config['head'], config['tail']))

    PART = 4
    TM = int(time())

    pool.apply_async(main, (NUM[0::PART], 621 + TM, 0))
    pool.apply_async(main, (NUM[1::PART], 928 + TM, 1))
    pool.apply_async(main, (NUM[2::PART], 122 + TM, 2))
    pool.apply_async(main, (NUM[3::PART], 222 + TM, 3))

    pool.close()
    pool.join()

    tmr.send('stop')

    print("Done.")
