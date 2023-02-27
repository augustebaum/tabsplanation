from experiments.cf_losses.task_create_plot_data_cf_losses import (
    TaskCreatePlotDataCfLosses,
)
from experiments.cf_losses.task_plot_cf_losses import TaskPlotCfLosses

from experiments.shared.data.task_get_data_module import TaskGetDataModule
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import clone_config


def parse_config(cfg):

    task_deps = [TaskPlotCfLosses(cfg)]
    task_deps.append(TaskCreatePlotDataCfLosses(cfg))

    # Make sure cfg is not modified
    cfg = clone_config(cfg)

    task_train_classifier = TaskTrainModel(cfg.classifier)
    task_train_autoencoder = TaskTrainModel(cfg.autoencoder)
    task_deps.append(task_train_classifier)
    task_deps.append(task_train_autoencoder)

    task_dataset = TaskGetDataModule.task_dataset(cfg.data_module)
    task_deps.append(task_dataset)

    tasks_to_collect = {}
    for task in task_deps:
        if task.__class__.__name__ not in tasks_to_collect:
            tasks_to_collect[task.__class__.__name__] = [task]
        else:
            tasks_to_collect[task.__class__.__name__].append(task)

    return tasks_to_collect
