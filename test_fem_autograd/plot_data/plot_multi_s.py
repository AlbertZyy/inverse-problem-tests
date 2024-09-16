
import os
from glob import glob
import json
from natsort import natsorted

from matplotlib import pyplot as plt


data_list = []
data_file_list = glob(os.path.join('test_fem_autograd/plot_data', 'multi_*.json'))


for data_file in data_file_list:
    with open(data_file, 'r') as f:
        data_list.append(json.load(f))

color_list = ['black', 'darkred', 'sienna', 'goldenrod', 'darkgreen', 'darkcyan', 'darkblue', 'indigo']

for data in data_list:
    if data['domain'] == 'full':
        continue

    noise = data['noise']
    fig = plt.figure(figsize=(8, 6))
    axes = fig.add_subplot(111)
    axes.set_title(f'$\delta = {noise}$', fontsize=16)
    idx = 0

    for idx, s0 in enumerate(data['s0']):
        ### plot a line for s0 ###
        x = [0, len(data['s_evo'][idx])]
        y = [s0, s0]
        axes.plot(x, y, color=color_list[idx], linewidth=0.8, linestyle='--')
        ### plot loss ###
        axes.plot(data['s_evo'][idx], color=color_list[idx], linewidth=0.8, label=r'$\gamma_{\text{true}}=$' + str(s0))

    axes.set_xlabel('iteration', fontsize=16)
    axes.set_ylabel('$\gamma$', fontsize=16)
    axes.legend(bbox_to_anchor=(0.5, 0.5), fontsize=10)

    plt.savefig(f'test_fem_autograd/figures/multi_s_noise_{str(noise).replace(".", "_")}.png')
