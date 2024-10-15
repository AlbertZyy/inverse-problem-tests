
import os
from glob import glob
import json

from matplotlib import pyplot as plt


data_list = []
data_file_list = glob(os.path.join('test_fem_autograd/plot_data', '*.json'))

for data_file in data_file_list:
    with open(data_file, 'r') as f:
        data_list.append(json.load(f))


noise_group = {}

for data in data_list:
    noise_group.setdefault(data['noise'], []).append(data)


color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

for noise, data_list in noise_group.items():
    sorted_by_s0 = sorted(data_list, key=lambda x: x['s0'])

    fig = plt.figure(figsize=(8, 6))
    axes = fig.add_subplot(111)
    axes.set_title(f'$\delta = {noise}$', fontsize=16)
    idx = 0

    for data in sorted_by_s0:
        if data['domain'] == 'full':
            continue

        ### plot a line for s0 ###
        x = [0, len(data['loss'])]
        y = [data['s0'], data['s0']]
        axes.plot(x, y, color=color_list[idx], linewidth=0.8, linestyle='--')

        ### plot loss ###
        axes.plot(data['loss'], color=color_list[idx], linewidth=0.8, label=r'$\gamma_{\text{true}}=$' + str(data["s0"]))
        axes.set_xlabel('iteration', fontsize=16)
        axes.set_ylabel('$\gamma$', fontsize=16)
        axes.legend(bbox_to_anchor=(0.5, 0.5), fontsize=14)

        plt.savefig(f'test_fem_autograd/figures/noise_{str(noise).replace(".", "_")}.png')
        idx += 1
