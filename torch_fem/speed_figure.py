
from matplotlib import pyplot as plt


fig = plt.figure(figsize=(8, 6))
axes = fig.add_subplot(111)

axes.plot(
    [16641, 66049, 263169, 1050625, 2362369],
    [0.16412, 0.45348, 1.73856, 4.04250, 8.47439],
    linewidth=0.8,
    marker='*',
    color='#0000ff'
)

axes.plot(
    [16641, 66049, 263169, 1050625, 2362369],
    [0.75906, 0.85730, 1.18855, 1.33297, 1.58196],
    linewidth=0.8,
    marker='*',
    color='#ff0000'
)

axes.plot(
    [16641, 66049, 263169, 1050625, 2362369],
    [0.02102, 0.13923, 2.20454, 26.56865, 168.72415],
    linewidth=0.8,
    linestyle='--',
    marker='*',
    color='#007fff'
)

axes.plot(
    [16641, 66049, 263169, 1050625, 2362369],
    [0.19710, 0.30185, 0.52189, 3.43641, 15.83130],
    linewidth=0.8,
    linestyle='--',
    marker='*',
    color='#ff7f00'
)

axes.set_yscale('log')
axes.set_xscale('log')
axes.set_ylabel('Time (s)', fontsize=16)
axes.set_xlabel('Degree of Freedom', fontsize=16)
axes.grid(True)
axes.legend(['Numpy Assemble', 'Torch Assemble', 'Numpy CG', 'Torch CG'], fontsize=14)

plt.show()
