
import os
from typing import Dict, Any, Optional

import argparse
import yaml
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Plot data from CSV files based on YAML configuration.')
    parser.add_argument('config', type=str, help='Path to the YAML configuration file.')
    return parser.parse_args()

def load_yaml_config(config_path) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_csv_data(file_path) -> Optional[DataFrame]:
    """使用 Pandas 读取 CSV 文件"""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def plot_data(config: Dict):
    """根据 YAML 配置绘制数据"""
    fig = plt.figure(figsize=config.get('figsize', (8, 6)))
    fig.suptitle(config.get('title', 'Visualize'))
    ax = fig.add_subplot(111)

    # Set log scales
    if config.get('xlog', False):
        ax.set_xscale('log')
    if config.get('ylog', False):
        ax.set_yscale('log')
    # Set axis limits
    if 'xlim' in config:
        ax.set_xlim(config['xlim'])
    if 'ylim' in config:
        ax.set_ylim(config['ylim'])
    # Set axis ticks
    if 'xticks' in config:
        ax.set_xticks(config['xticks'])
    if 'yticks' in config:
        ax.set_yticks(config['yticks'])
    # Set axis labels
    ax.set_xlabel(config.get('xlabel', 'X'), fontsize=14)
    ax.set_ylabel(config.get('ylabel', 'Y'), fontsize=14)

    for file_path, plot_info in config['plot'].items():
        data = load_csv_data(file_path)
        if data is not None:
            xkey = plot_info['xkey']
            ykey = plot_info['ykey']
            plot_args = plot_info.get('args', {})

            # 使用提供的参数绘制图形
            ax.plot(data[xkey], data[ykey], **plot_args)
        else:
            print(f"Warning: Data could not be loaded from file '{file_path}'.")

    ax.legend(fontsize=14)

    ### add geometry
    if 'rectangle' in config.keys():
        ax.add_patch(plt.Rectangle(**config['rectangle']))

    if 'output' in config:
        os.makedirs(os.path.dirname(config['output']), exist_ok=True)
        plt.savefig(config['output'])

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    args = parse_arguments()
    config = load_yaml_config(args.config)
    plot_data(config)
