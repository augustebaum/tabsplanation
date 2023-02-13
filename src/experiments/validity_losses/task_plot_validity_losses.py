from config import BLD_PLOTS

from experiments.shared.utils import get_object, read, setup, Task, write

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
        import pdb

        pdb.set_trace()
        pass
