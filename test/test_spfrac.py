
import sys
sys.path.append("./src")

import torch

from fractional import SparkleFractional


frac = SparkleFractional(252, 4, dtype=torch.float64)
frac.from_npz(r"./data/laplace_beltrami_63_63.npz")

frac.initialize(s0=0.5)

print(frac.s, frac.s0)

frac.sparkle()

print(frac.s, frac.s0)

frac.initialize(s=[0.1, 0.2, 0.3, 0.4])

print(frac.s, frac.s0)

frac.shrink()

print(frac.s, frac.s0)
