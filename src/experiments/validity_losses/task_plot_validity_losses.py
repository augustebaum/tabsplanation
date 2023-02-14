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

        # method 1
        # xx = df.pivot_table(
        #     columns=["data_module", "path_method"],
        #     index="loss",
        #     aggfunc=["mean", "sem"],
        # )

        # method 2
        yy = df.groupby(list(df.columns[:-1])).agg(["mean", "sem"])
        yy = yy.droplevel(0, axis=1)

        # import numpy as np

        # def highlight_max(x):
        #     return np.where(x == np.nanmax(x.to_numpy()), "textbf", None)

        # yy = yy.style.highlight_max(
        #     color=None, props="font-weight: bold;", subset="mean"
        # )
        # yy = yy.style.highlight_max(color=None, props="textbf:;", subset="mean")

        # yy = yy.apply(
        #     lambda row: ("{:.4f}".format(row["mean"]), "{:.4f}".format(row["sem"])),
        #     axis=1,
        # )
        yy = yy.apply(
            lambda row: "{:.4f}".format(row["mean"])
            + "±"
            + "{:.4f}".format(row["sem"]),
            axis=1,
        )

        yy = yy.unstack([0, 1])

        # yy = yy.apply(
        #     lambda row: "{:.4f}".format(row["mean"])
        #     + "±"
        #     + "{:.4f}".format(row["sem"]),
        #     axis=1,
        # )

        # yy.style.highlight_max(color=None, props="bfseries: ;")
        write(yy.style.to_latex(), produces["results"])

        # df = pd.DataFrame(np.random.randn(5, 2), columns=["A", "B"])
        # df.style.apply(highlight_max, color='red')
        # df.style.apply(highlight_max, color='blue', axis=1)
        # df.style.apply(highlight_max, color='green', axis=None)

        # write("", produces)
        # pass
