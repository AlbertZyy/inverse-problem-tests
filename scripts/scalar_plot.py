
import argparse
from typing import Dict, Any

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import yaml


def exponential_moving_average(time_series: NDArray, alpha: float):
    """
    Calculate the exponential moving average (EMA) of a time series.

    Parameters:
    - time_series (numpy.ndarray): Input time series data.
    - alpha (float): Smoothing factor, between 0 and 1.

    Returns:
    - numpy.ndarray: Exponential moving average of the input time series.
    """

    ema = np.zeros_like(time_series)
    ema[0] = time_series[0]

    for i in range(1, len(time_series)):
        ema[i] = (1 - alpha) * time_series[i] + alpha * ema[i - 1]

    return ema


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


fig = plt.figure(config['fig_name'])
axes = fig.subplots(1, 1)


for file_name in config['csv_files']:
    data = np.loadtxt(file_name, delimiter=',', skiprows=1)

    step = data[:, 1].astype(np.int_)
    value = data[:, 2]

    if config.get('log', False):
        value = np.log10(value)

    if config.get('smooth', 0):
        value = exponential_moving_average(value, config['smooth'])

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

lim_args, lim_kwargs = set_lim(config.get('xlim', 'auto'))
axes.set_xlim(*lim_args, **lim_kwargs)
lim_args, lim_kwargs = set_lim(config.get('ylim', 'auto'))
axes.set_ylim(*lim_args, **lim_kwargs)

if config['save']:
    fig.savefig(config['save'])

if config['show']:
    plt.show()
