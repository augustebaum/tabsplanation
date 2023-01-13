import pickle

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap

from config import BLD_PLOTS
from experiments.cf_losses.task_create_plot_data_cf_losses import (
    get_z0,
    TaskCreatePlotDataCfLosses,
)
from experiments.shared.utils import define_task, load_mpl_style, Task


class TaskPlotCfLosses(Task):
    """So far assume the inputs are 2-dimensional."""

    def __init__(self, cfg):
        output_dir = BLD_PLOTS / "cf_losses"
        super(TaskPlotCfLosses, self).__init__(cfg, output_dir)

        self.depends_on = TaskCreatePlotDataCfLosses(cfg).produces

        self.produces |= {"plot": self.produces_dir / "plot.svg"}

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        with open(depends_on["results"], "rb") as results_file:
            results = pickle.load(results_file)

        z0 = get_z0(cfg)
        z0, z1 = torch.meshgrid(z0, z0)
        nb_classes = 3

        load_mpl_style()
        fig, axes = plt.subplots(
            nrows=nb_classes, ncols=len(results), layout="constrained"
        )

        for class_ in range(nb_classes):
            axes[class_, 0].annotate(
                f"Target class is {class_}",
                xy=(0, 0.5),
                xytext=(-axes[class_, 0].yaxis.labelpad - 5, 0),
                xycoords=axes[class_, 0].yaxis.label,
                textcoords="offset points",
                size="large",
                ha="right",
                va="center",
            )

        for i, (loss_name, loss_data) in enumerate(results.items()):
            for class_ in range(nb_classes):
                ax = axes[class_][i]
                cs = ax.contourf(
                    z0,
                    z1,
                    loss_data[class_].detach().reshape((cfg.nb_steps, cfg.nb_steps)),
                    zorder=1,
                    cmap=LinearSegmentedColormap.from_list("", ["white", "red"], N=256),
                    norm=plt.Normalize(),
                )

                ax.axis([cfg.lo, cfg.hi, cfg.lo, cfg.hi])
                ax.set_title(loss_name)
                ax.set_xlabel(r"$z_0$")
                ax.set_ylabel(r"$z_1$")

        fig.colorbar(cs)
        plt.show(block=True)


task, task_definition = define_task("cf_losses", TaskPlotCfLosses)
exec(task_definition)
