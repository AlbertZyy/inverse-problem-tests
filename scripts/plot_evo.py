
import argparse
from typing import Dict, Any

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import yaml


def exponential_moving_average(time_series: NDArray, alpha: float):
    assert time_series.ndim == 1
    ema = 0.0
    biased = np.zeros_like(time_series)

    for i in range(len(time_series)):
        ema = (1 - alpha) * time_series[i] + alpha * ema
        biased[i] = ema / (1 - alpha**(i+1))

    return biased


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
    config: Dict[str, Any] = yaml.load(f.read(), Loader=yaml.FullLoader)


fig = plt.figure(config['fig_name'], figsize=config['fig_size'])
axes = fig.subplots(1, 1)

legends = []

for legend, data in config['data'].items():
    legends.append(legend)
    file_name = data['csv_file']
    evo_data = np.loadtxt(file_name, delimiter=',', skiprows=1)
    step = evo_data[:, 1].astype(np.int_)
    value = evo_data[:, 2]

    if config.get('smooth', 0):
        value = exponential_moving_average(value, config['smooth'])

    step = step[config['index']]
    value = value[config['index']]

    if config.get('log', False):
        axes.set_yscale('log')

    axes.plot(step, value,
              marker=data['marker'],
              color=data['color'],
              linestyle=data['linestyle'],
              label=legend)


axes.legend(legends)
axes.set_xlabel(config['xlabel'])
axes.set_ylabel(config['ylabel'])
axes.set_title(config['title'])
axes.grid()

lim_args, lim_kwargs = set_lim(config.get('xlim', 'auto'))
axes.set_xlim(*lim_args, **lim_kwargs)
lim_args, lim_kwargs = set_lim(config.get('ylim', 'auto'))
axes.set_ylim(*lim_args, **lim_kwargs)

if lim_args:
    if 'ytick_num' in config.keys():
        axes.set_yticks(np.linspace(*lim_args, num=config['ytick_num']))

if config['save']:
    fig.savefig(config['save'])

if config['show']:
    plt.show()
