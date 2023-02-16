import pandas as pd

from config import BLD_PLOTS

from experiments.shared.utils import parse_full_qualified_object, read, Task, write
from experiments.validity_losses.task_create_plot_data_validity_losses import (
    TaskCreatePlotDataValidityLosses,
)


class TaskPlotValidityLosses(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOTS / "validity_losses"
        super(TaskPlotValidityLosses, self).__init__(cfg, output_dir)

        task_plot_data = TaskCreatePlotDataValidityLosses(cfg)
        self.task_deps.append(task_plot_data)
        self.depends_on = task_plot_data.produces

        self.produces |= {"results": self.produces_dir / "results.tex"}

    @staticmethod
    def parse_result(result):
        get_object_name = lambda s: parse_full_qualified_object(s)[1]
        return {
            "Dataset": get_object_name(result["data_module"]).removesuffix("Dataset"),
            "Path method": result["path_method"]["name"],
            "Loss function": get_object_name(result["loss"]),
            "validity_rate": result["validity_rate"],
        }

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        results = read(depends_on["results"])
        results = [TaskPlotValidityLosses.parse_result(result) for result in results]

        df = pd.DataFrame.from_records(results)

        df = df.groupby(list(df.columns[:-1])).agg(["mean", "sem"])
        df = df * 100
        df = df.droplevel(0, axis=1)

        df = df.apply(
            lambda row: "{:.1f}".format(row["mean"])
            + "±"
            + "{:.1f}".format(row["sem"]),
            axis=1,
        )

        df = df.unstack([0, 1])

        loss_names_in_order = [r["Loss function"] for r in results][: len(cfg.losses)]
        df = df.reindex(loss_names_in_order)

        write(df.style.to_latex(), produces["results"])
