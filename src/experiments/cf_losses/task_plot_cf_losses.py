import pickle

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap

from config import BLD_PLOTS
from experiments.cf_losses.task_create_plot_data_cf_losses import (
    get_x0,
    get_z0,
    Gradients,
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
            "x_gradients": self.produces_dir / "x_gradients.svg",
            "z_gradients": self.produces_dir / "z_gradients.svg",
        }

    @classmethod
    def task_function(cls, depends_on, produces, cfg):

        with open(depends_on["results"], "rb") as results_file:
            results = pickle.load(results_file)

        load_mpl_style()

        # 1. Losses in input space
        fig = TaskPlotCfLosses.plot_loss_contours(
            results["x_losses"],
            x_axis=get_x0(),
            axis_limits=[-5, 55, -5, 55],
            axis_labels=(r"$x_0$", r"$x_1$"),
            title="Loss in input space",
        )
        for ax in fig.axes:
            if "colorbar" not in ax.get_label():
                ax.imshow(
                    get_map_img(), origin="upper", extent=[0, 50, 0, 50], zorder=2
                )

        fig.savefig(produces["x_losses"])

        # 2. Losses in latent space
        fig = TaskPlotCfLosses.plot_loss_contours(
            results["z_losses"],
            x_axis=get_z0(),
            axis_limits=[-5, 5, -5, 5],
            axis_labels=(r"$z_0$", r"$z_1$"),
            title="Loss in latent space",
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

        # 3.a. Gradients in input space with respect to x
        fig = TaskPlotCfLosses.plot_gradients(
            results["x_gradients"],
            x_axis=get_x0(),
            axis_limits=[-5, 55, -5, 55],
            axis_labels=(r"$x_0$", r"$x_1$"),
            title=r"Opposite of gradient with respect to $x$",
        )

        fig.savefig(produces["x_gradients"])

        # 3.b. Gradients in input space with respect to x
        fig = TaskPlotCfLosses.plot_gradients(
            results["z_gradients"],
            x_axis=get_x0(),
            axis_limits=[-5, 55, -5, 55],
            axis_labels=(r"$x_0$", r"$x_1$"),
            title=r"Opposite of gradient with respect to $z$",
        )

        fig.savefig(produces["z_gradients"])

        plt.show(block=True)

        fig, ax = plt.subplots()
        ax.scatter(
            z[:, 0],
            z[:, 1],
            c=results["latent_space_map"]["class"].detach(),
            alpha=0.5,
            marker="s",
            zorder=1,
        )

    @staticmethod
    def plot_latent_space_map(ax, latent_space_map):
        z = latent_space_map["z"]
        class_ = latent_space_map["class"]

        ax.scatter(
            z[:, 0],
            z[:, 1],
            c=class_.detach(),
            alpha=0.5,
            marker="s",
            zorder=1,
        )

    @staticmethod
    def plot_loss_contours(
        results: ResultDict[Loss], x_axis, axis_limits, axis_labels, title
    ):
        x0, x1 = torch.meshgrid(x_axis, x_axis)
        nb_classes = len(results[list(results.keys())[0]])

        fig, axes = plt.subplots(
            nrows=nb_classes, ncols=len(results), layout="constrained", figsize=(12, 10)
        )

        fig.suptitle(title)

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

    @staticmethod
    def plot_gradients(
        results: ResultDict[Gradients], x_axis, axis_limits, axis_labels, title
    ):
        x0, x1 = torch.meshgrid(x_axis, x_axis, indexing="xy")
        nb_classes = len(results[list(results.keys())[0]])

        fig, axes = plt.subplots(
            nrows=nb_classes, ncols=len(results), layout="constrained", figsize=(12, 10)
        )

        fig.suptitle(title)

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

        for col, loss_name in enumerate(results.keys()):
            for row, class_ in enumerate(range(nb_classes)):

                ax = axes[row][col]

                gradients = results[loss_name][class_]

                u, v = (
                    gradients[:, 0].reshape((len(x0), len(x0))).T.detach(),
                    gradients[:, 1].reshape((len(x0), len(x0))).T.detach(),
                )

                ax.streamplot(x0.numpy(), x1.numpy(), u.numpy(), v.numpy())

                ax.axis(axis_limits)
                ax.set_xlabel(axis_labels[0])
                ax.set_ylabel(axis_labels[1])
                if row == 0:
                    ax.set_title(loss_name)

                ax.imshow(get_map_img(), origin="upper", extent=[0, 50, 0, 50])

        return fig


task, task_definition = define_task("cf_losses", TaskPlotCfLosses)
exec(task_definition)
