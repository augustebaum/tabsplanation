from omegaconf import OmegaConf

from config import EXPERIMENT_CONFIGS

from experiments.validity_losses.task_plot_validity_losses import TaskPlotValidityLosses

cfg = OmegaConf.load(EXPERIMENT_CONFIGS / "validity_losses.yaml")
task = TaskPlotValidityLosses(cfg)
task_def = task.define_task()
exec(task_def)
