
import numpy as np

data = np.load(r"./data/laplace_beltrami_63_63.npz")
v = data["v"]
w = data["w"]
M = data["M"]

print(v.shape, w.shape, M.shape)

print(v.T@M@v)
