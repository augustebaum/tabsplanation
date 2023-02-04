import pickle

import pandas as pd

from config import BLD_PLOTS
from experiments.compare_cf_path_methods.task_create_plot_data_compare_cf_methods import (
    TaskCreatePlotDataCfPathMethods,
)
from experiments.shared.utils import Task


class TaskPlotMethodStats(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOTS / "cf_path_methods" / "method_stats"
        super(TaskPlotMethodStats, self).__init__(cfg, output_dir)

        task_create_plot_data_cf_path_methods = TaskCreatePlotDataCfPathMethods(
            self.cfg
        )
        self.depends_on = task_create_plot_data_cf_path_methods.produces

        self.produces |= {
            "raw_stats": self.produces_dir / "raw_stats.csv",
            "latex_friendly_stats": self.produces_dir / "latex_friendly_stats.csv",
        }
        print(f"Raw stats would be saved in \n{self.produces['raw_stats']}")

    @classmethod
    def task_function(cls, depends_on, produces, cfg):

        with open(depends_on["results"], "rb") as results_file:
            results = pickle.load(results_file)

        validity_df = pd.DataFrame(
            [
                [path["validity"] for path in results[method.class_name]]
                for method in cfg.methods
            ]
        ).T
        validity_df.columns = [method.class_name for method in cfg.methods]
        validity_rate: pd.Series = validity_df.mean()

        runtime_df = pd.DataFrame(
            [
                [
                    path["runtime_per_step_milliseconds"]
                    for path in results[method.class_name]
                ]
                for method in cfg.methods
            ]
        ).T
        runtime_df.columns = [method.class_name for method in cfg.methods]
        time_stats_df = runtime_df.agg(["mean", "sem"]).T.add_suffix("_time_ms")

        stats_df = time_stats_df.copy()
        stats_df["validity_rate"] = validity_rate

        stats_df.to_csv(produces["raw_stats"])

        latex_stats_df = stats_df.copy()
        latex_stats_df["Runtime per step (ms)"] = (
            latex_stats_df["mean_time_ms"].map("{:,.2f}".format)
            + "$\pm$"
            + latex_stats_df["sem_time_ms"].map("{:,.2f}".format)
        )
        latex_stats_df["Validity rate"] = latex_stats_df["validity_rate"].map(
            "{:,.1%}".format
        )

        latex_stats_df[["Runtime per step (ms)", "Validity rate"]].to_csv(
            produces["latex_friendly_stats"]
        )
