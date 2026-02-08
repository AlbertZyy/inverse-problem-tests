from time import time
import numpy as np
import torch

from fealpy.backend import backend_manager as bm
from fealpy.sparse import csr_matrix

bm.set_backend('numpy')

if bm.backend_name == 'pytorch':
    bm.set_default_device('cuda:4')


file = np.load("data/cfd1.npz")
# file = np.load("data/poisson3Da.npz")
data = bm.from_numpy(file["data"])
indptr = bm.from_numpy(file["indptr"])
indices = bm.from_numpy(file["indices"])
shape = file["shape"]

total_time = 0.
LOOPS = 50

# A = csr_matrix((data, indices, indptr), shape=shape)
A = torch.sparse_csr_tensor(indptr, indices, data, shape)
SIZE = A.shape[1]
print(A)

for i in range(LOOPS + 1): # the first one is warm up
    F = bm.random.rand(SIZE)
    F = bm.astype(F, bm.float64)
    t0 = time()
    u = A.matmul(F)
    t1 = time()

    if i > 0:
        total_time += t1 - t0

print(f"{total_time/LOOPS * 1000} ms")


# NOTE: Results

### poisson3Da
"""
NumPy, CPU: 0.29282093048095703 ms
PyTorch, CUDA: 0.015521049499511717 ms
"""

### cfd1
"""
NumPy, CPU: 1.3202428817749023 ms
PyTorch, CUDA: 0.021429061889648438 ms
"""