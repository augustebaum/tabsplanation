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

    task_deps = [TaskPlotPathRegularization(cfg)]

    task_deps.append(TaskCreatePlotDataPathRegularization(cfg))

    # Make sure cfg is not modified
    cfg = clone_config(cfg)
    full_config = []

    setup(cfg.seed)
    seeds = [random.randrange(100_000) for _ in range(cfg.nb_seeds)]

    # For each dataset
    for data_module_cfg in cfg.data_modules:
        # Get the DataModule
        task_dataset = TaskGetDataModule.task_dataset(data_module_cfg)
        task_deps.append(task_dataset)

        # Train a classifier
        classifier_cfg = cfg.classifier
        classifier_cfg.data_module = data_module_cfg
        task_classifier = TaskTrainModel(classifier_cfg)

        # Add it to the dependencies
        task_deps.append(task_classifier)
        # data_module_name = data_module_cfg.dataset.class_name
        # depends_on[data_module_name] = {
        #     "dataset": task_dataset.produces,
        #     "classifier": task_classifier.produces,
        #     "autoencoders": {},
        # }

        # Short-hand
        # autoencoder_deps = depends_on[data_module_name]["autoencoders"]

        for seed in seeds:

            # Train an unregularized autoencoder
            autoencoder_cfg = cfg.autoencoder
            autoencoder_cfg.data_module = data_module_cfg
            autoencoder_cfg.seed = seed
            task_autoencoder = TaskTrainModel(autoencoder_cfg)

            # Add it to the dependencies
            task_deps.append(task_autoencoder)
            # Path regularized = False
            # autoencoder_deps[(seed, False)] = task_autoencoder.produces
            full_config.append(
                {
                    "data_module": data_module_cfg,
                    "classifier": classifier_cfg,
                    "autoencoder": autoencoder_cfg,
                    "path_regularized": False,
                }
            )

            for explainer_cfg in cfg.explainers:
                path_regularized_autoencoder_cfg = clone_config(autoencoder_cfg)
                path_regularized_autoencoder_cfg.model = {
                    "class_name": "PathRegularizedNICE",
                    "args": {
                        "classifier": classifier_cfg,
                        "autoencoder_args": autoencoder_cfg.model.args,
                        "explainer": explainer_cfg,
                    },
                }

                # Train a regularized autoencoder with the same architecture
                task_path_regularized_autoencoder = TaskTrainPathRegAe(
                    path_regularized_autoencoder_cfg
                )

                # Add it to the dependencies
                task_deps.append(task_path_regularized_autoencoder)
                # Path regularized = True
                # autoencoder_deps[
                #     (seed, True)
                # ] = task_path_regularized_autoencoder.produces

                full_config.append(
                    {
                        "data_module": data_module_cfg,
                        "classifier": classifier_cfg,
                        "autoencoder": autoencoder_cfg,
                        "path_regularized": explainer_cfg,
                    }
                )

    tasks_to_collect = {}
    for task in task_deps:
        if task.__class__.__name__ not in tasks_to_collect:
            tasks_to_collect[task.__class__.__name__] = [task]
        else:
            tasks_to_collect[task.__class__.__name__].append(task)

    return tasks_to_collect
