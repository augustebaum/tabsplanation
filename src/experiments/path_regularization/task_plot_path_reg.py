"""
Draft task to start plotting results.

The graphs are not super informative but it's a first step.
"""


import matplotlib.pyplot as plt

from config import BLD_PLOTS
from experiments.cf_losses.task_plot_cf_losses import TaskPlotCfLosses
from experiments.latent_shift.task_plot_class_2_paths import TaskPlotClass2Paths
from experiments.path_regularization.task_create_plot_data_path_reg import (
    TaskCreatePlotDataPathReg,
)
from experiments.shared.utils import define_task, load_mpl_style, read, Task, write


class TaskPlotPathReg(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOTS / "path_reg"
        super(TaskPlotPathReg, self).__init__(cfg, output_dir)

        self.depends_on = TaskCreatePlotDataPathReg(self.cfg).produces

        self.produces |= {
            "latent_space_maps": self.produces_dir / "latent_space_maps.svg",
            "test_paths": self.produces_dir / "test_paths.svg",
        }

    @classmethod
    def task_function(cls, depends_on, produces, cfg):

        results = read(depends_on["results"])

        load_mpl_style()

        # Plot latent space maps
        fig, ax = plt.subplots(ncols=2, squeeze=True)

        TaskPlotCfLosses.plot_latent_space_map(
            ax[0], results["unregularized_latent_space_map"]
        )
        ax[0].set_title("Unregularized")

        TaskPlotCfLosses.plot_latent_space_map(
            ax[1], results["path_regularized_latent_space_map"]
        )
        ax[1].set_title("Path regularized")

        write(fig, produces["latent_space_maps"])

        # Plot paths
        fig, ax = plt.subplots(ncols=3, squeeze=True)

        for i, (method_name, path) in enumerate(results["paths"].items()):
            TaskPlotClass2Paths.plot_path(ax[i], path)
            ax[i].set_title(method_name)

        write(fig, produces["test_paths"])

        plt.show(block=True)


task, task_definition = define_task("path_reg", TaskPlotPathReg)
exec(task_definition)
