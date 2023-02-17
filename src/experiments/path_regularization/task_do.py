from omegaconf import OmegaConf

from config import EXPERIMENT_CONFIGS

from experiments.path_regularization.task_plot_path_regularization import (
    TaskPlotPathRegularization,
)

cfg = OmegaConf.load(EXPERIMENT_CONFIGS / "path_regularization.yaml")
task = TaskPlotPathRegularization(cfg)
task_def = task.define_task()
exec(task_def)
