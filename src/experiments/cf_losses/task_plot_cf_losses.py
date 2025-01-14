import pickle

import matplotlib.pyplot as plt
import torch

from config import BLD_PLOTS
from experiments.cf_losses.task_create_plot_data_cf_losses import (
    get_x0,
    get_z0,
    Gradients,
    Loss,
    ResultDict,
    TaskCreatePlotDataCfLosses,
)
from experiments.shared.utils import get_map_img, load_mpl_style, Task


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
        z = results["latent_space_map"]["z"].detach().cpu()

        fig, ax = plt.subplots()
        ax.scatter(
            z[:, 0],
            z[:, 1],
            c=results["latent_space_map"]["class"].detach().cpu(),
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
        TaskPlotCfLosses.plot_latent_space_map(ax, results["latent_space_map"])

    @staticmethod
    def plot_latent_space_map(ax, latent_space_map):
        z = latent_space_map["z"].detach().cpu()
        class_ = latent_space_map["class"].detach().cpu()

        ax.scatter(
            z[:, 0],
            z[:, 1],
            c=class_,
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

        figsize = (nb_classes * 5, len(results) * 4)
        fig, axes = plt.subplots(
            nrows=len(results), ncols=nb_classes, layout="constrained", figsize=figsize
        )

        fig.suptitle(title)

        # Row labels (class number)
        put_row_labels(axes, [loss_name for loss_name in results.keys()])

        for row, (loss_name, loss_data) in enumerate(results.items()):
            for col, class_ in enumerate(range(nb_classes)):

                ax = axes[row][col]

                cs = ax.contourf(
                    x0.cpu().numpy(),
                    x1.cpu().numpy(),
                    loss_data[class_]
                    .detach()
                    .reshape((len(x0), len(x0)))
                    .cpu()
                    .numpy(),
                    zorder=1,
                    norm=plt.Normalize(),
                )

                ax.axis(axis_limits)
                ax.set_xlabel(axis_labels[0])
                ax.set_ylabel(axis_labels[1])
                if row == 0:
                    ax.set_title(f"target = {class_}")

            fig.colorbar(cs, ax=ax, location="bottom")

        return fig

    @staticmethod
    def plot_gradients(
        results: ResultDict[Gradients], x_axis, axis_limits, axis_labels, title
    ):
        x0, x1 = torch.meshgrid(x_axis, x_axis, indexing="xy")
        nb_classes = len(results[list(results.keys())[0]])

        figsize = (nb_classes * 5, len(results) * 4)
        fig, axes = plt.subplots(
            nrows=len(results), ncols=nb_classes, layout="constrained", figsize=figsize
        )

        fig.suptitle(title)

        # Row labels (class number)
        put_row_labels(axes, [loss_name for loss_name in results.keys()])

        for row, (loss_name, loss_data) in enumerate(results.items()):
            for col, class_ in enumerate(range(nb_classes)):

                ax = axes[row][col]
                if row == 0:
                    ax.set_title(f"target = {class_}")

                gradients = results[loss_name][class_]

                u, v = (
                    gradients[:, 0].reshape((len(x0), len(x0))).T.detach(),
                    gradients[:, 1].reshape((len(x0), len(x0))).T.detach(),
                )

                ax.streamplot(
                    x0.cpu().numpy(), x1.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy()
                )

                ax.axis(axis_limits)
                ax.set_xlabel(axis_labels[0])
                ax.set_ylabel(axis_labels[1])

                ax.imshow(get_map_img(), origin="upper", extent=[0, 50, 0, 50])

        return fig


def put_row_labels(axes, labels):
    for i, label in enumerate(labels):
        ax = axes[i, 0]
        ax.annotate(
            label,
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
        )
