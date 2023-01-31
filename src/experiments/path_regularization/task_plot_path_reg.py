"""
Draft task to start plotting results.

The graphs are not super informative but it's a first step.
"""

from typing import List

import matplotlib.pyplot as plt

from config import BLD_PLOTS
from experiments.cf_losses.task_plot_cf_losses import TaskPlotCfLosses
from experiments.latent_shift.task_plot_class_2_paths import TaskPlotClass2Paths
from experiments.path_regularization.task_create_plot_data_path_reg import (
    TaskCreatePlotDataPathRegularization,
)
from experiments.shared.utils import define_task, load_mpl_style, read, Task, write


# <https://stackoverflow.com/questions/8598163/split-title-of-a-figure-in-matplotlib-into-multiple-lines>
# def split_title_line(title_text, split_on="(", max_words=3):
#     """
#     A function that splits any string based on specific character
#     (returning it with the string), with maximum number of words on it
#     """
#     split_at = title_text.find(split_on)
#     ti = title_text
#     if split_at > 1:
#         ti = ti.split(split_on)
#         for i, tx in enumerate(ti[1:]):
#             ti[i + 1] = split_on + tx
#     if type(ti) == type("text"):
#         ti = [ti]
#     for j, td in enumerate(ti):
#         if td.find(split_on) > 0:
#             pass
#         else:
#             tw = td.split()
#             t2 = []
#             for i in range(0, len(tw), max_words):
#                 t2.append(" ".join(tw[i : max_words + i]))
#             ti[j] = t2
#     ti = [item for sublist in ti for item in sublist]
#     ret_tex = []
#     for j in range(len(ti)):
#         for i in range(0, len(ti) - 1, 2):
#             if len(ti[i].split()) + len(ti[i + 1].split()) <= max_words:
#                 mrg = " ".join([ti[i], ti[i + 1]])
#                 ti = [mrg] + ti[2:]
#                 break

#     if len(ti[-2].split()) + len(ti[-1].split()) <= max_words:
#         mrg = " ".join([ti[-2], ti[-1]])
#         ti = ti[:-2] + [mrg]
#     return "\n".join(ti)


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

        self.depends_on = TaskCreatePlotDataPathRegularization(self.cfg).produces

        self.produces |= {
            "latent_space_maps": self.produces_dir / "latent_space_maps.svg",
            "test_paths": self.produces_dir / "test_paths.svg",
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

        plt.show(block=True)


task, task_definition = define_task("path_reg", TaskPlotPathReg)
exec(task_definition)
