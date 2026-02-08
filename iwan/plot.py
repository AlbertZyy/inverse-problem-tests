
from sucrose.post import LogDataFrame, plot_evolution
from matplotlib import pyplot as plt


RUNS = ["simple/alpha"]

fig = plt.figure("iwan")
axes = fig.add_subplot(1, 1, 1)

df = LogDataFrame(
    "iwan",
    runs=RUNS,
    tags=["normA", "normB1", "normB2", "normB3", "normBPhi1", "normBPhi2"]
).load()
plot_evolution(df, axes, ["k-", "b-", "r-", "g-", "c-", "y-"], log_scale="y")

axes.grid()
axes.legend()
fig.savefig("iwan/figures/" + "_".join(RUNS).replace("/", "_") + ".png")
