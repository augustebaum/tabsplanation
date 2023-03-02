import os

from pprint import pprint

import pytask
from omegaconf import OmegaConf

from config import BLD_RUNS
from experiments.shared.utils import read, save_config, save_full_config


# Read latest task file
tasks_to_collect = read(BLD_RUNS / sorted(os.listdir(BLD_RUNS))[-1] / "tasks.pkl")

pprint(tasks_to_collect)

for task_list in tasks_to_collect.values():
    for task in task_list:
        task_def = task._define_task()
        exec(task_def)
