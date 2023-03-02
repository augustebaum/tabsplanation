import random
from typing import Dict, List

from experiments.path_regularization.task_create_plot_data_path_regularization import (
    TaskCreatePlotDataPathRegularization,
)

from experiments.path_regularization.task_plot_path_regularization import (
    TaskPlotPathRegularization,
)
from experiments.path_regularization_cake_on_sea.task_train_path_regularized_ae import (
    TaskTrainPathRegAe,
)
from experiments.shared.data.task_get_data_module import TaskGetDataModule
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import clone_config, setup, Task

TaskName = str
TaskDict = Dict[TaskName, List[Task]]


def parse_config(cfg) -> TaskDict:
    task = TaskPlotPathRegularization(cfg)
    task_deps = [task] + task.all_task_deps()

    tasks_to_collect = {}
    for task in task_deps:
        if task.__class__.__name__ not in tasks_to_collect:
            tasks_to_collect[task.__class__.__name__] = [task]
        else:
            tasks_to_collect[task.__class__.__name__].append(task)

    return tasks_to_collect
