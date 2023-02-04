from omegaconf import OmegaConf

from config import EXPERIMENT_CONFIGS

from experiments.validity_losses.task_create_plot_data_validity_losses import (
    TaskCreatePlotDataValidityLosses,
)

cfg = OmegaConf.load(EXPERIMENT_CONFIGS / "validity_losses.yaml")
task = TaskCreatePlotDataValidityLosses(cfg)
task, task_def = task.define_task()
exec(task_def)
