import pickle

import matplotlib.pyplot as plt

from config import BLD_PLOTS
from experiments.compare_cf_path_methods.task_create_plot_data_compare_cf_methods import (
    TaskCreatePlotDataCfPathMethods,
)
from experiments.latent_shift.task_plot_class_2_paths import TaskPlotClass2Paths
from experiments.shared.utils import define_task, load_mpl_style, Task


class TaskPlotCfPathMethods(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOTS / "cf_path_methods"
        super(TaskPlotCfPathMethods, self).__init__(cfg, output_dir)

        task_create_plot_data_cf_path_methods = TaskCreatePlotDataCfPathMethods(
            self.cfg
        )
        self.depends_on = task_create_plot_data_cf_path_methods.produces

        self.produces |= {"plot": self.produces_dir / "plot.svg"}

    @classmethod
    def task_function(cls, depends_on, produces, cfg):

        with open(depends_on["results"], "rb") as results_file:
            results = pickle.load(results_file)

        load_mpl_style()
        nrows, ncols = len(cfg.methods), 3
        figsize = (3 * ncols, 8 / 4 * nrows)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        ax[0, 0].set_title(r"Likelihood of perturbation $\uparrow$")
        ax[0, 1].set_title(r"Distance to explained input $\downarrow$")
        ax[0, 2].set_title(r"Paths")

        for i, method in enumerate(cfg.methods):

            ax[i, 0].annotate(
                method.class_name,
                xy=(0, 0.5),
                xytext=(-ax[i, 0].yaxis.labelpad - 5, 0),
                xycoords=ax[i, 0].yaxis.label,
                textcoords="offset points",
                size="large",
                ha="right",
                va="center",
            )

            method_results = results[method.class_name]
            for result in method_results:
                ax[i, 0].plot(result["likelihoods_nf"])
                ax[i, 0].set_xlabel("Iterations")
                ax[i, 0].set_ylabel("NF Likelihood")

                ax[i, 1].plot(result["l1_distances_to_input"])
                ax[i, 1].set_xlabel("Iterations")
                ax[i, 1].set_ylabel("$L_1$ distance")

                TaskPlotClass2Paths.plot_path(ax[i, 2], result["path"])

        # plt.show(block=True)
        fig.savefig(produces["plot"])


define_task("compare_cf_methods", TaskPlotCfPathMethods)
