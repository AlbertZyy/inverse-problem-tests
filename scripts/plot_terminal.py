
import argparse
from typing import Dict, Any

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import yaml


def ema_terminal(time_series: NDArray, alpha: float) -> float:
    assert time_series.ndim == 1
    ema = 0.0
    L = len(time_series)

    for value in time_series:
        ema: float = (1 - alpha) * value + alpha * ema

    return ema / (1 - alpha**(L+1))


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
    ends = []
    if 'csv_files' in data:
        for fname in data.pop('csv_files'):
            evo_data = np.loadtxt(fname, delimiter=',', skiprows=1)
            step = evo_data[:, 1].astype(np.int_)
            value = evo_data[:, 2]

            if config.get('smooth', 0):
                ending = ema_terminal(value, config['smooth'])
            else:
                ending = value[-1]

            ends.append(ending)
    elif 'values' in data:
        for v in data.pop('values'):
            ends.append(v)
    else:
        raise ValueError(f"Can not find data for legend {legend}.")

    x = config.get('x', None)
    if x is None:  # x is step
        x = range(len(ends))
    axes.plot(x, ends, label=legend, **data)

if config.get('xlog', False) is True:  # xlog
    axes.set_xscale('log')
if config.get('ylog', False) is True:
    axes.set_yscale('log')

axes.legend(legends, fontsize=18)
axes.set_xlabel(config['xlabel'], fontsize=18)
axes.set_ylabel(config['ylabel'], fontsize=18)
axes.set_title(config['title'], fontsize=18)
axes.grid()

lim_args, lim_kwargs = set_lim(config.get('xlim', 'auto'))
axes.set_xlim(*lim_args, **lim_kwargs)
xticks = config.get('xticks', None)
plt.xticks(**xticks, fontsize=14) if xticks is not None else plt.xticks(fontsize=14)

lim_args, lim_kwargs = set_lim(config.get('ylim', 'auto'))
axes.set_ylim(*lim_args, **lim_kwargs)
if lim_args:
    if 'ytick_num' in config.keys():
        tick_data = np.linspace(*lim_args, num=config['ytick_num'])
        tick_str = [f'{v:.2f}' for v in tick_data]
        axes.set_yticks(ticks=tick_data, labels=tick_str, fontsize=14)

if config['save']:
    fig.savefig(config['save'])

if config['show']:
    plt.show()
