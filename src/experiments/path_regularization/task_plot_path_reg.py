import math
from typing import List, Tuple

import matplotlib.pyplot as plt

from config import BLD_PLOTS, EXPERIMENT_CONFIGS
from experiments.cf_losses.task_plot_cf_losses import TaskPlotCfLosses
from experiments.latent_shift.task_plot_class_2_paths import TaskPlotClass2Paths
from experiments.path_regularization.task_create_plot_data_path_reg import (
    TaskCreatePlotDataPathRegularization,
)
from experiments.shared.utils import load_mpl_style, read, Task, write
from tabsplanation.types import Tensor


def split_title_line(text: str, max_words=3):
    words = text.split(" ")

    lines: List[str] = []
    for i in range(0, len(words), max_words):
        lines.append(" ".join(words[i : max_words + i]))

    return "\n".join(lines)


class TaskPlotPathReg(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOTS / "path_reg"
        super(TaskPlotPathReg, self).__init__(cfg, output_dir)

        task_create_plot_data_path_regularization = (
            TaskCreatePlotDataPathRegularization(self.cfg)
        )
        self.task_deps = [task_create_plot_data_path_regularization]

        self.depends_on = task_create_plot_data_path_regularization.produces
        self.depends_on |= {"config": EXPERIMENT_CONFIGS / "path_reg.yaml"}

        self.produces |= {
            "latent_space_maps": self.produces_dir / "latent_space_maps.svg",
            "test_paths": self.produces_dir / "test_paths.svg",
            "z_gradients": self.produces_dir / "z_gradients.svg",
        }

    @classmethod
    def task_function(cls, depends_on, produces, cfg):

        results = read(depends_on["results"])

        load_mpl_style()

        # Plot latent space maps
        fig, ax = plt.subplots(ncols=2, squeeze=True)

        TaskPlotCfLosses.plot_latent_space_map(
            ax[0], results["unregularized_latent_space_map"]
        )
        ax[0].set_title("Unregularized")

        TaskPlotCfLosses.plot_latent_space_map(
            ax[1], results["path_regularized_latent_space_map"]
        )
        ax[1].set_title("Path regularized")

        write(fig, produces["latent_space_maps"])

        # Plot paths
        fig, ax = plt.subplots(ncols=3, squeeze=True)

        for i, (method_name, path) in enumerate(results["paths"].items()):
            TaskPlotClass2Paths.plot_path(ax[i], path)
            ax[i].set_title(split_title_line(method_name))

        write(fig, produces["test_paths"])

        # Plot gradients
        fig = TaskPlotPathReg.plot_latent_grads_map(results["gradients"])

        write(fig, produces["z_gradients"])

        plt.show(block=True)

    @staticmethod
    def plot_latent_grads_map(grad_results):
        def to_numpy(
            tensor: Tensor["nb_points ** 2", 2]
        ) -> Tuple[Tensor["nb_points", "nb_points"], Tensor["nb_points", "nb_points"]]:
            nb_points = round(math.sqrt(len(tensor)))
            return (
                tensor[:, 0].reshape((nb_points, nb_points)).T.detach().cpu().numpy(),
                tensor[:, 1].reshape((nb_points, nb_points)).T.detach().cpu().numpy(),
            )

        fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(10, 10))

        for col, (autoencoder_name, target_class_results) in enumerate(
            grad_results.items()
        ):
            ax[0][col].set_title(autoencoder_name)

            for row, (target_class, grad_results) in enumerate(
                target_class_results.items()
            ):

                # for i, (autoencoder_name, grad_results) in enumerate(
                #     results["gradients"].items()
                # ):
                z = grad_results["latents"].cpu()
                grads_z = grad_results["grads_z"].cpu()

                z0, z1 = to_numpy(z)
                u, v = to_numpy(grads_z)

                ax[row][col].streamplot(z0, z1, u, v)

                ax[row][col].scatter(
                    z[:, 0],
                    z[:, 1],
                    c=grad_results["classes"].cpu(),
                    alpha=0.2,
                    marker="s",
                    zorder=1,
                )

                if col == 0:
                    ax[target_class, 0].annotate(
                        f"Target class is {target_class}",
                        xy=(0, 0.5),
                        xytext=(-ax[target_class, 0].yaxis.labelpad - 5, 0),
                        xycoords=ax[target_class, 0].yaxis.label,
                        textcoords="offset points",
                        size="large",
                        ha="right",
                        va="center",
                    )

        return fig


from omegaconf import OmegaConf

cfg = OmegaConf.load(EXPERIMENT_CONFIGS / "path_reg.yaml")
task, task_def = TaskPlotPathReg(cfg).define_task()
exec(task_def)
