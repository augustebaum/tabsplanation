import pandas as pd

from config import BLD_PLOTS
from experiments.robustness_path_regularization.task_create_plot_data_robustness_path_regularization import (
    TaskCreatePlotDataRobustnessPathRegularization,
)

from experiments.shared.utils import read, Task, write


class TaskPlotRobustnessPathRegularization(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOTS / "robustness_path_regularization"
        super(TaskPlotRobustnessPathRegularization, self).__init__(cfg, output_dir)

        task_plot_data = TaskCreatePlotDataRobustnessPathRegularization(cfg)
        self.task_deps.append(task_plot_data)
        self.depends_on = task_plot_data.produces

        self.produces |= {"results": self.produces_dir / "results.tex"}

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        results = read(depends_on["results"])
        import pdb

        pdb.set_trace()

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
