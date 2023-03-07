from typing import Dict, List

from experiments.robustness_path_regularization.task_plot_robustness_path_regularization import (
    TaskPlotRobustnessPathRegularization,
)
from experiments.shared.utils import Task

TaskName = str
TaskDict = Dict[TaskName, List[Task]]


def parse_config(cfg) -> TaskDict:
    task = TaskPlotRobustnessPathRegularization(cfg)
    task_deps = [task] + task.all_task_deps()

    tasks_to_collect = {}
    for task in task_deps:
        if task.__class__.__name__ not in tasks_to_collect:
            tasks_to_collect[task.__class__.__name__] = [task]
        else:
            tasks_to_collect[task.__class__.__name__].append(task)

    return tasks_to_collect
