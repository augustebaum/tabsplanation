import pickle

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap

from config import BLD_PLOTS
from experiments.cf_losses.task_create_plot_data_cf_losses import (
    get_x0,
    get_z0,
    Loss,
    ResultDict,
    TaskCreatePlotDataCfLosses,
)
from experiments.shared.utils import define_task, get_map_img, load_mpl_style, Task


class TaskPlotCfLosses(Task):
    """So far assume the inputs are 2-dimensional."""

    def __init__(self, cfg):
        output_dir = BLD_PLOTS / "cf_losses"
        super(TaskPlotCfLosses, self).__init__(cfg, output_dir)

        self.depends_on = TaskCreatePlotDataCfLosses(cfg).produces

        self.produces |= {
            "x_losses": self.produces_dir / "x_losses.svg",
            "z_losses": self.produces_dir / "z_losses.svg",
            "latent_space_map": self.produces_dir / "latent_space_map.svg",
        }

    @classmethod
    def task_function(cls, depends_on, produces, cfg):

        with open(depends_on["results"], "rb") as results_file:
            results = pickle.load(results_file)

        load_mpl_style()

        # 1. Losses in input space
        fig = TaskPlotCfLosses.plot_all_losses_and_targets(
            results["x_losses"],
            x_axis=get_x0(),
            axis_limits=[-5, 55, -5, 55],
            axis_labels=(r"$x_0$", r"$x_1$"),
        )
        # ax.imshow(get_map_img(), origin="upper", extent=[0, 50, 0, 50])

        fig.savefig(produces["x_losses"])

        # 2. Losses in latent space
        fig = TaskPlotCfLosses.plot_all_losses_and_targets(
            results["z_losses"],
            x_axis=get_z0(),
            axis_limits=[-5, 5, -5, 5],
            axis_labels=(r"$z_0$", r"$z_1$"),
        )

        fig.savefig(produces["z_losses"])

        # 2.b. Class distribution in latent space
        z = results["latent_space_map"]["z"].detach()

        fig, ax = plt.subplots()
        ax.scatter(
            z[:, 0],
            z[:, 1],
            c=results["latent_space_map"]["class"].detach(),
            alpha=0.5,
            marker="s",
            zorder=1,
        )

        fig.savefig(produces["latent_space_map"])

        # plt.show(block=True)

    @staticmethod
    def plot_all_losses_and_targets(
        results: ResultDict[Loss], x_axis, axis_limits, axis_labels
    ):
        x0, x1 = torch.meshgrid(x_axis, x_axis)
        nb_classes = len(results[list(results.keys())[0]])

        fig, axes = plt.subplots(
            nrows=nb_classes, ncols=len(results), layout="constrained", figsize=(12, 10)
        )

        # Row labels (class number)
        for class_ in range(nb_classes):
            axes[class_, 0].annotate(
                f"Target class is {class_}",
                xy=(0, 0.5),
                xytext=(-axes[class_, 0].yaxis.labelpad - 5, 0),
                xycoords=axes[class_, 0].yaxis.label,
                textcoords="offset points",
                size="large",
                ha="right",
                va="center",
            )

        for col, (loss_name, loss_data) in enumerate(results.items()):
            for row, class_ in enumerate(range(nb_classes)):

                ax = axes[row][col]

                cs = ax.contourf(
                    x0,
                    x1,
                    loss_data[class_].detach().reshape((len(x0), len(x0))),
                    zorder=1,
                    cmap=LinearSegmentedColormap.from_list("", ["white", "red"]),
                    norm=plt.Normalize(),
                )

                ax.axis(axis_limits)
                ax.set_xlabel(axis_labels[0])
                ax.set_ylabel(axis_labels[1])
                if row == 0:
                    ax.set_title(loss_name)

            fig.colorbar(cs, ax=ax, location="bottom")

        return fig


task, task_definition = define_task("cf_losses", TaskPlotCfLosses)
exec(task_definition)
