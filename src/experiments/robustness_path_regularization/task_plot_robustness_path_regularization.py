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

        df = pd.DataFrame.from_records(results)

        df = df.groupby(list(df.columns[:4])).agg(["mean", "sem"])

        formats = {
            (r"Validity rate (\%)", "mean"): "{:.1f}",
            (r"Validity rate (\%)", "sem"): "{:.1f}",
            (r"\Delta t (ns)", "mean"): "{:.1f}",
            (r"\Delta t (ns)", "sem"): "{:.1f}",
            ("Mean NLL", "mean"): "{:.1E}",
            ("Mean NLL", "sem"): "{:.1E}",
            ("Mean distance to max", "mean"): "{:.2f}",
            ("Mean distance to max", "sem"): "{:.2f}",
        }
        for col, format_str in formats.items():
            df[col] = df[col].apply(format_str.format)

        df = df.stack([0, 1]).unstack([1, 2, 3, 5])

        write(df.style.to_latex(), produces["results"])
