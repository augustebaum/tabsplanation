import pickle

import matplotlib.pyplot as plt
import pytask
from matplotlib.collections import LineCollection
from matplotlib.figure import Axes

from config import BLD_PLOTS
from experiments.latent_shift.task_create_plot_data_class_2_paths import (
    TaskCreatePlotDataClass2Paths,
)
from experiments.shared.utils import get_configs, get_map_img, hash_, save_config, setup
from tabsplanation.types import LatentShiftPath


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

    @staticmethod
    def _plot_name(i: int) -> str:
        return f"path_{i:03d}"


cfgs = get_configs("latent_shift")

for cfg in cfgs:
    task = TaskPlotClass2Paths(cfg)

    @pytask.mark.task(id=task.id_)
    @pytask.mark.depends_on(task.depends_on)
    @pytask.mark.produces(task.produces)
    def task_plot_class_2_paths(depends_on, produces, cfg=task.cfg):

        setup(cfg.seed)

        with open(depends_on["paths"], "rb") as paths_file:
            paths = pickle.load(paths_file)

        fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")

        for i, path in enumerate(paths, start=1):
            _plot_path(ax, path)

            fig.savefig(produces[TaskPlotClass2Paths._plot_name(i)])
            plt.cla()

        save_config(cfg, produces["config"])


def _plot_path(ax: Axes, path: LatentShiftPath):

    margin = 5
    ax.axis([0 - margin, 50 + margin, 0 - margin, 50 + margin])

    ax.imshow(get_map_img(), origin="upper", extent=[0, 50, 0, 50])

    # Show explained input
    x = path.explained_input.x
    ax.scatter(x[0], x[1], c="red")

    # Show CF path
    cfs = path.xs.squeeze()[:, [0, 1]]
    ax.add_collection(
        LineCollection(segments=[cfs], linestyles="dashed", linewidths=1, label="cf")
    )

    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_1$")
