
import numpy as np

data = np.load(r"./data/laplace_beltrami_63_63.npz")
v = data["v"]
w = data["w"]
vinv = data["vinv"]

print(v.shape, w.shape, vinv.shape)
print(vinv@v)

from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
cm = ax.imshow(vinv@v)
fig.colorbar(cm)
plt.show()
