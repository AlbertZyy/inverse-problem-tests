
import argparse

import numpy as np
import matplotlib.pyplot as plt
import yaml


def set_lim(conf: str):
    if isinstance(conf, str):
        if conf in {'auto', 'Auto'}:
            return (), {'auto': True}
        else:
            raise ValueError(f"Unknown limit type {conf}.")

    elif isinstance(conf, list):
        return (conf[0], conf[1]), {}


parser = argparse.ArgumentParser(description='Plot scalar data')
parser.add_argument('input_file', type=str, help='file to plot')

args = parser.parse_args()


with open(args.input_file, 'r') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)


fig = plt.figure(config['fig_name'])
axes = fig.subplots(1, 1)


for file_name in config['csv_files']:
    data = np.loadtxt(file_name, delimiter=',', skiprows=1)

    step = data[:, 1].astype(np.int_)
    value = data[:, 2]

    if config['log']:
        value = np.log10(value)

    if config['plot_type'] in {'line', 'Line', 'L'}:
        axes.plot(step, value)

    elif config['plot_type'] in {'scatter', 'Scatter', 'S'}:
        axes.scatter(step, value)

    else:
        raise ValueError(f"Unknown plot type {config['plot_type']}.")


axes.legend(config['legends'])
axes.set_xlabel(config['xlabel'])
axes.set_ylabel(config['ylabel'])
axes.set_title(config['title'])
axes.grid()

lim_args, lim_kwargs = set_lim(config['xlim'])
axes.set_xlim(*lim_args, **lim_kwargs)
lim_args, lim_kwargs = set_lim(config['ylim'])
axes.set_ylim(*lim_args, **lim_kwargs)

if config['save']:
    fig.savefig(config['save'])

if config['show']:
    plt.show()
