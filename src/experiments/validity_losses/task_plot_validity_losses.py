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

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        results = read(depends_on["results"])
        results = [
            {
                "data_module": parse_full_qualified_object(result["data_module"])[1],
                "path_method": result["path_method"]["name"],
                "loss": parse_full_qualified_object(result["loss"])[1],
                "validity_rate": result["validity_rate"],
            }
            for result in results
        ]

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

        write(df.style.to_latex(), produces["results"])
