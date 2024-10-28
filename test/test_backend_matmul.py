
from time import time
from fealpy.backend import backend_manager as bm

bm.set_backend('numpy')

if bm.backend_name == 'pytorch':
    bm.set_default_device('cuda:4')


SIZE = 300
total_time = 0.
LOOPS = 50

for i in range(LOOPS + 1): # the first one is warm up
    A = bm.random.rand(SIZE, SIZE)
    F = bm.random.rand(SIZE, SIZE)
    A = bm.astype(A, bm.float64)
    F = bm.astype(F, bm.float64)
    t0 = time()
    u = bm.matmul(A, F)
    t1 = time()

    if i > 0:
        total_time += t1 - t0

print(f"{total_time/LOOPS * 1000} ms")

# NOTE: the result is 
# SIZE = 300
"""
NumPy, CPU: 0.9653663635253906 ms
PyTorch, CUDA: 0.02391815185546875 ms
"""

# SIZE = 3000
"""
NumPy, CPU: 172.27213859558105 ms
PyTorch, CUDA: 0.020322799682617188 ms
"""