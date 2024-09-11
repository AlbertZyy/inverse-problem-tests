
from typing import Dict
import os
import csv

import tensorflow as tf
from tqdm import tqdm


StepData = Dict[int, Dict[str, float]]


def read_scalar_events(run_path: str, *, tqdm_=False, position: int = 0) -> StepData:
    """
    Read 目录中的tfevents文件，并返回一个字典，
    字典的键是tag，值是一个列表，包含每个step对应的值。
    """
    all_steps_data = {}
    event_file_list = os.listdir(run_path)
    LEN = len(event_file_list)

    for idx, event_file in enumerate(event_file_list):

        if event_file.startswith('events.out.tfevents'):
            event_file_path = os.path.join(run_path, event_file)
            event_iter = tf.compat.v1.train.summary_iterator(event_file_path)

            if tqdm_:
                event_iter = tqdm(event_iter, desc=f'Reading Events ({idx}/{LEN})',
                                  ascii=True, leave=True, position=position)

            for e in event_iter:
                for v in e.summary.value:
                    if v.HasField('simple_value'):  # 只处理标量类型
                        tag = v.tag
                        step = e.step
                        value = v.simple_value
                        if step not in all_steps_data:
                            all_steps_data[step] = {}
                        all_steps_data[step][tag] = value

    return all_steps_data


def write_to_csv(steps_data: StepData, output_file: str):
    """
    将按步排序的标量数据写入CSV文件。
    """
    if not steps_data:
        return

    tags = set().union(*steps_data.values())
    tags = sorted(tags)

    with open(output_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['Step'] + tags
        writer.writerow(header)

        for step in sorted(steps_data.keys()):
            row = [step] + [steps_data[step].get(tag, None) for tag in tags]
            writer.writerow(row)


def process_logs(log_dir: str, output_dir: str, *, tqdm_=False):
    """
    处理log目录下的所有训练记录，并为每个训练记录生成一个CSV文件。
    """
    os.makedirs(output_dir, exist_ok=True)
    run_dir_iter = os.listdir(log_dir)

    if tqdm_:
        run_dir_iter = tqdm(run_dir_iter, desc='Processing', ascii=True)

    for run_dir in run_dir_iter:
        run_path = os.path.join(log_dir, run_dir)

        if os.path.isdir(run_path):
            steps_data = read_scalar_events(run_path, tqdm_=tqdm_, position=1)
            output_file = os.path.join(output_dir, f"{run_dir}.csv")
            write_to_csv(steps_data, output_file)
            print(f"Processed {run_dir} -> {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str)
    parser.add_argument("output_dir", type=str, default="output")
    args = parser.parse_args()

    log_dir = args.log_dir
    output_dir = args.output_dir

    process_logs(log_dir, output_dir, tqdm_=True)
