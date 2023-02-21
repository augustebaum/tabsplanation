import random

from experiments.shared.data.task_get_data_module import TaskGetDataModule
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import clone_config, setup

from experiments.validity_losses.task_create_plot_data_validity_losses import (
    TaskCreatePlotDataValidityLosses,
)


def parse_config(cfg):

    task_deps = [TaskCreatePlotDataValidityLosses(cfg)]
    # Make sure cfg is not modified
    cfg = clone_config(cfg)
    full_config = []

    setup(cfg.seed)
    seeds = [random.randrange(100_000) for _ in range(cfg.nb_seeds)]

    # Make sure cfg is not modified
    cfg = clone_config(cfg)

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

        # Then for each seed
        for seed in seeds:
            # Train an autoencoder
            autoencoder_cfg = cfg.autoencoder
            autoencoder_cfg.data_module = data_module_cfg
            autoencoder_cfg.seed = seed
            task_autoencoder = TaskTrainModel(autoencoder_cfg)

            # Add it to the dependencies
            task_deps.append(task_autoencoder)
            # autoencoder_deps[seed] = task_autoencoder.produces

            # full_config.append(
            #     {
            #         "data_module": data_module_cfg,
            #         "classifier": classifier_cfg,
            #         "autoencoder": autoencoder_cfg,
            #     }
            # )

    tasks_to_collect = {}
    for task in task_deps:
        if task.__class__.__name__ not in tasks_to_collect:
            tasks_to_collect[task.__class__.__name__] = [task]
        else:
            tasks_to_collect[task.__class__.__name__].append(task)

    return tasks_to_collect
