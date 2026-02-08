
from sucrose.post import LogDataFrame, plot_evolution
from matplotlib import pyplot as plt


RUNS = ["base2"]

fig = plt.figure("ann")
axes = fig.add_subplot(1, 1, 1)

df = LogDataFrame(
    "ann",
    runs=RUNS,
    tags=["loss(train)", "loss(validate)"]
).load()
plot_evolution(df, axes, ["r-", "b-"], log_scale="xy", legend_fmt="{tag}")

axes.grid()
axes.legend()
# axes.set_title("validation loss")
fig.savefig("ann/figures/" + "_and_".join(RUNS).replace("/", "_") + ".png")
