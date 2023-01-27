import pickle

import matplotlib.pyplot as plt
import pytask
from matplotlib.collections import LineCollection
from matplotlib.figure import Axes

from config import BLD_PLOTS
from experiments.latent_shift.task_create_plot_data_class_2_paths import (
    TaskCreatePlotDataClass2Paths,
)
from experiments.shared.utils import (
    get_configs,
    get_map_img,
    hash_,
    load_mpl_style,
    save_config,
    save_full_config,
)
from tabsplanation.types import ExplanationPath


class TaskPlotClass2Paths:
    def __init__(self, cfg):
        self.cfg = cfg

        task_create_plot_data_class_2_paths = TaskCreatePlotDataClass2Paths(self.cfg)

        self.depends_on = task_create_plot_data_class_2_paths.produces

        self.id_ = hash_(self.cfg)
        plots_dir = BLD_PLOTS / "class_2_paths" / self.id_

        n = cfg.plot_data_class_2_paths.nb_points
        plot_names = [TaskPlotClass2Paths._plot_name(i) for i in range(1, n + 1)]

        self.produces = {name: plots_dir / f"{name}.svg" for name in plot_names}
        self.produces["config"] = plots_dir / "config.yaml"
        self.produces["full_config"] = plots_dir / "full_config.yaml"

    def task_function(depends_on, produces, cfg):

        with open(depends_on["paths"], "rb") as paths_file:
            paths = pickle.load(paths_file)

        load_mpl_style()
        fig, ax = plt.subplots(figsize=(5, 5))

        for i, path in enumerate(paths, start=1):
            TaskPlotClass2Paths.plot_path(ax, path)

            fig.savefig(produces[TaskPlotClass2Paths._plot_name(i)])

            plt.cla()

    @staticmethod
    def _plot_name(i: int) -> str:
        return f"path_{i:03d}"

    @staticmethod
    def plot_path(ax: Axes, path: ExplanationPath):

        margin = 5
        ax.axis([0 - margin, 50 + margin, 0 - margin, 50 + margin])

        ax.imshow(get_map_img(), origin="upper", extent=[0, 50, 0, 50])

        # Show explained input
        x = path.explained_input.x.squeeze()
        ax.scatter(x[0], x[1], c="red")

        # Show CF path
        cfs = path.xs.reshape(-1, len(x))[:, [0, 1]]
        ax.add_collection(
            LineCollection(
                segments=[cfs], linestyles="dashed", linewidths=1, label="cf"
            )
        )

        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel(r"$x_1$")


_task_class = TaskPlotClass2Paths
cfgs = get_configs("latent_shift")

for cfg in cfgs:
    task = _task_class(cfg)

    @pytask.mark.task(id=task.id_)
    @pytask.mark.depends_on(task.depends_on)
    @pytask.mark.produces(task.produces)
    def task_plot_class_2_paths(depends_on, produces, cfg=task.cfg):
        _task_class.task_function(depends_on, produces, cfg)
        save_config(cfg, produces["config"])
        save_full_config(cfg, produces["full_config"])
