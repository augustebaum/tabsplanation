import pickle

import matplotlib.pyplot as plt

from config import BLD_PLOTS
from experiments.compare_cf_path_methods.task_create_plot_data_compare_cf_methods import (
    TaskCreatePlotDataCfPathMethods,
)
from experiments.shared.utils import load_mpl_style, Task


def make_boxplots(
    depends_on: dict,
    produces: dict,
    cfg_methods: dict,
    plot_title: str,
    metric_name: str,
):
    with open(depends_on["results"], "rb") as results_file:
        results = pickle.load(results_file)

    load_mpl_style()
    fig, ax = plt.subplots()

    ax.set_title(plot_title)
    ax.boxplot(
        [
            [result[metric_name] for result in results[method.class_name]]
            for method in cfg_methods
        ],
        labels=[method.class_name for method in cfg_methods],
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.savefig(produces["plot"])
    # plt.show(block=True)


class TaskPlotAucLof(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOTS / "cf_path_methods" / "auc_lof"
        super(TaskPlotAucLof, self).__init__(cfg, output_dir)

        task_create_plot_data_cf_path_methods = TaskCreatePlotDataCfPathMethods(
            self.cfg
        )
        self.depends_on = task_create_plot_data_cf_path_methods.produces

        self.produces |= {"plot": self.produces_dir / "plot.svg"}
        print(f"Plot would be saved in \n{self.produces['plot']}")

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        plot_title = "AUC of LOF for each method"
        metric_name = "lof_auc"

        make_boxplots(depends_on, produces, cfg.methods, plot_title, metric_name)
