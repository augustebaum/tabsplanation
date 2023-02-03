from config import BLD_PLOTS
from experiments.compare_cf_path_methods.task_create_plot_data_compare_cf_methods import (
    TaskCreatePlotDataCfPathMethods,
)
from experiments.compare_cf_path_methods.task_plot_compare_cf_methods_auc_lof import (
    make_boxplots,
)
from experiments.shared.utils import define_task, Task


class TaskPlotAucL1(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOTS / "cf_path_methods" / "auc_l1"
        super(TaskPlotAucL1, self).__init__(cfg, output_dir)

        task_create_plot_data_cf_path_methods = TaskCreatePlotDataCfPathMethods(
            self.cfg
        )
        self.depends_on = task_create_plot_data_cf_path_methods.produces

        self.produces |= {"plot": self.produces_dir / "plot.svg"}
        print(f"Plot would be saved in \n{self.produces['plot']}")

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        plot_title = r"AUC of $L_1$ distance for each method"
        metric_name = "l1_distances_to_input_auc"

        make_boxplots(depends_on, produces, cfg.methods, plot_title, metric_name)


# task, task_definition = define_task("compare_cf_methods", TaskPlotAucL1)
# exec(task_definition)
