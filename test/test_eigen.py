
import numpy as np

data = np.load(r"./data/laplace_beltrami_63_63.npz")
v = data["v"]
w = data["w"]
vinv = data["vinv"]

print(v.shape, w.shape, vinv.shape)
print(np.linalg.norm(vinv@v - np.eye(252), 2))

from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
cm = ax.imshow(vinv@v - np.eye(252))
fig.colorbar(cm)
plt.show()
