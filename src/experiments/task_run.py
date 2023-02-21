# import pdb

import pytask
from omegaconf import OmegaConf

import experiments.run
from experiments.shared.utils import save_config, save_full_config

# pdb.set_trace()

print(experiments.run.tasks_to_collect)
for task_list in experiments.run.tasks_to_collect.values():
    for task in task_list:
        task_def = task._define_task()
        exec(task_def)
