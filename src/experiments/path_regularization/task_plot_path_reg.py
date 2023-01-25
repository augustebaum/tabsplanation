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
from experiments.shared.utils import define_task, load_mpl_style, read, Task


class TaskPlotPathReg(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOTS / "path_reg"
        super(TaskPlotPathReg, self).__init__(cfg, output_dir)

        self.depends_on = TaskCreatePlotDataPathReg(self.cfg).produces

        self.produces |= {"plot": self.produces_dir / "plot.svg"}
        print(f"Plot would be saved in \n{self.produces['plot']}")

    @classmethod
    def task_function(cls, depends_on, produces, cfg):

        results = read(depends_on["results"])

        load_mpl_style()

        # Plot latent space maps
        fig, ax = plt.subplots(nrows=1, ncols=2)
        TaskPlotCfLosses.plot_latent_space_map(
            ax[0, 0], results["unregularized_latent_space_map"]
        )
        ax[0, 0].set_title("Unregularized")
        TaskPlotCfLosses.plot_latent_space_map(
            ax[0, 1], results["path_regularized_latent_space_map"]
        )
        ax[0, 1].set_title("Path regularized")

        # nrows, ncols = len(cfg.methods), 4
        # figsize = (3 * ncols, 8 / 4 * nrows)

        # ax[0, 0].set_title(r"Likelihood of perturbation $\uparrow$")
        # ax[0, 1].set_title(r"Distance to explained input $\downarrow$")
        # ax[0, 2].set_title(r"Paths")
        # ax[0, 3].set_title(r"Local Outlier Factor $\uparrow$")

        # for i, method in enumerate(cfg.methods)

        #     ax[i, 0].annotate(
        #         method.class_name,
        #         xy=(0, 0.5),
        #         xytext=(-ax[i, 0].yaxis.labelpad - 5, 0),
        #         xycoords=ax[i, 0].yaxis.label,
        #         textcoords="offset points",
        #         size="large",
        #         ha="right",
        #         va="center",
        #     )

        #     method_results = results[method.class_name]
        #     for result in method_results:
        #         ax[i, 0].plot(result["likelihoods_nf"])
        #         ax[i, 0].set_xlabel("Iterations")
        #         ax[i, 0].set_ylabel("NF Likelihood")

        #         ax[i, 1].plot(result["l1_distances_to_input"])
        #         ax[i, 1].set_xlabel("Iterations")
        #         ax[i, 1].set_ylabel("$L_1$ distance")

        #         TaskPlotClass2Paths.plot_path(ax[i, 2], result["path"])

        #         ax[i, 3].plot(result["lof"])
        #         ax[i, 3].set_xlabel("Iterations")
        #         ax[i, 3].set_ylabel("LOF")

        # plt.show(block=True)

        fig.savefig(produces["plot"])

        plt.show(block=True)


task, task_definition = define_task("compare_cf_methods", TaskPlotCfPathMethods)
exec(task_definition)
