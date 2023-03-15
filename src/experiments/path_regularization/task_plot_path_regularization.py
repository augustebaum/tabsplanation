import pandas as pd

from config import BLD_PLOTS
from experiments.path_regularization.task_create_plot_data_path_regularization import (
    TaskCreatePlotDataPathRegularization,
)

from experiments.shared.utils import read, Task, write


class TaskPlotPathRegularization(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOTS / "path_regularization"
        super(TaskPlotPathRegularization, self).__init__(cfg, output_dir)

        task_plot_data = TaskCreatePlotDataPathRegularization(cfg)
        self.task_deps.append(task_plot_data)
        self.depends_on = task_plot_data.produces

        self.produces |= {"results": self.produces_dir / "results.tex"}

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        results = read(depends_on["results"])

        df = pd.DataFrame.from_records(results)

        df = df.groupby(list(df.columns[:4])).agg(["mean", "sem"])

        formats = {
            ("validity_rate (%)", "mean"): "{:.1f}",
            ("validity_rate (%)", "sem"): "{:.1f}",
            (r"\Delta t (ns)", "mean"): "{:.1f}",
            (r"\Delta t (ns)", "sem"): "{:.1f}",
            ("Mean #BC", "mean"): "{:.2f}",
            ("Mean #BC", "sem"): "{:.2f}",
            ("Mean NLL", "mean"): "{:.1E}",
            ("Mean NLL", "sem"): "{:.1E}",
        }
        for col, format_str in formats.items():
            df[col] = df[col].apply(format_str.format)

        write(df.style.to_latex(), produces["results"])
