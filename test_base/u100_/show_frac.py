
from unet_100 import build_model

model, MODEL_NAME = build_model('cpu', tag='N')

frac = model.df_solver._frac
gain = frac.gain

from matplotlib import pyplot as plt

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
axes.plot(gain.T.detach().abs().numpy())
# axes.set_ylim(-0.1, 0.1)

# axes = fig.add_subplot(1, 2, 2)
# axes.plot(alpha_2.T.detach().abs().numpy())
# axes.set_ylim(0, 2.)

plt.show()
