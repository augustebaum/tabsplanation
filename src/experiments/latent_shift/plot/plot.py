from typing import Iterable, List

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.figure import Axes
from torchtyping import TensorType

from tabsplanation.explanations import LatentShiftPath, make_path
from tabsplanation.types import Shift


def get_probabilities_data(explanations: List[LatentShiftPath]):

    shifts: List[List[Shift]] = [e.shifts for e in explanations]
    prb_old_class = [e.prbs_old for e in explanations]
    prb_new_class = [e.prbs_new for e in explanations]

    return shifts, prb_old_class, prb_new_class


def probabilities_plot(
    ax: Axes,
    shifts: List[TensorType["nb_explanations", 1]],
    prb_old_class: List[TensorType["nb_explanations", 1]],
    prb_new_class: List[TensorType["nb_explanations", 1]],
) -> None:
    """Plot the prediction probabilities of counterfactuals,
    specifically for the class that was originally predicted,
    and for desired target class.
    """
    ax.axis([-1.05, 1.05, -1.05, 1.05])

    line_params = dict(color="k", alpha=0.2, linestyle="dotted", zorder=1e3)
    ax.axhline(y=0, xmin=-1.05, xmax=1.05, **line_params)
    ax.axvline(x=0, ymin=-1.05, ymax=1.05, **line_params)

    def line_collection(shifts, values, color, label):
        segments = [torch.cat((s, v), dim=1) for s, v in zip(shifts, values)]
        colors = mcolors.to_rgba(color, alpha=0.2)
        line_widths = 2

        return LineCollection(
            segments=segments, colors=colors, linewidths=line_widths, label=label
        )

    ax.add_collection(
        line_collection(
            shifts, prb_old_class, color="b", label=r"$\Delta p^\mathrm{(old\ class)}$"
        )
    )
    ax.add_collection(
        line_collection(
            shifts, prb_new_class, color="r", label=r"$\Delta p^\mathrm{(new\ class)}$"
        )
    )

    ax.set_xlabel(r"Latent shift ($\lambda / \lambda_\mathrm{max}$)")
    ax.set_ylabel(r"Probability change ($p_\mathrm{CF}^{(c)} - p^{(c)}$)")

    ax.legend()


def get_map_data(explanations: List[LatentShiftPath]):
    # Take first two features
    xs = torch.cat(
        [e.explained_input.x[[0, 1]].reshape(-1, 2) for e in explanations], axis=0
    )
    cfs = [e.xs[:, [0, 1]] for e in explanations]
    return xs, cfs


def get_map_img():
    return mpimg.imread("experiments/data_map.png")


def map_plot(ax: Axes, paths: Iterable[LatentShiftPath]):

    margin = 5
    ax.axis([0 - margin, 50 + margin, 0 - margin, 50 + margin])

    img = get_map_img()
    ax.imshow(img, origin="upper", extent=[0, 50, 0, 50])

    ax.set_prop_cycle(color=["red", "green", "blue"])
    xs = torch.stack([path.explained_input.x for path in paths])
    ax.scatter(xs[:, 0], xs[:, 1])

    # Reset color cycle to reuse the colors of the scattered points
    ax.set_prop_cycle(color=["red", "green", "blue"])
    cfs = [path.xs for path in paths]
    ax.add_collection(
        LineCollection(segments=cfs, linestyles="dashed", linewidths=1, label="cf")
    )

    # Add markers for each CF (can't do that directly in LineCollection)
    # print(cfs)
    # cf_points = torch.cat(cfs)
    # print(cf_points)
    # ax.scatter(cf_points[:, 0], cf_points[:, 1], s=10, c="k")

    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_1$")


def plot_experiment(inputs, target_map, aes, clfs, normalize_inverse):
    """

    Inputs:
    -------
    target_map: For each possible output class, determines what should be
        the target class, e.g. `{0: 1, 1: None, 2: 1}`.
    """
    proba_plot_args = [[None for _ in range(len(clfs))] for _ in range(len(aes))]
    map_plot_args = [[None for _ in range(len(clfs))] for _ in range(len(aes))]
    for row, ae in enumerate(aes):
        for col, clf in enumerate(clfs):

            explanations = []

            for input in inputs:

                predicted_class = np.argmax(clf(input).detach()).item()
                print("predicted_class", predicted_class)

                explanations.append(
                    make_path(
                        input=input,
                        target_class=target_map[predicted_class],
                        clf=clf,
                        ae=ae,
                    )
                )

            # De-normalize all points in the input space
            for explanation in explanations:
                explanation.explained_input.input = normalize_inverse(
                    explanation.explained_input.input
                )

                explanation.xs = normalize_inverse(explanation.xs)

            proba_plot_args[row][col] = get_probabilities_data(explanations)
            map_plot_args[row][col] = get_map_data(explanations)

    fig = plt.figure(layout="constrained")
    figs = fig.subfigures(nrows=len(aes), ncols=len(clfs), squeeze=False)
    for row, ae in enumerate(aes):
        for col, clf in enumerate(clfs):
            subfig = figs[row][col]
            ax = subfig.subplots(1, 2, squeeze=True)

            probabilities_plot(ax[0], *proba_plot_args[row][col])
            map_plot(ax[1], *map_plot_args[row][col])

            if row == 0:
                subfig.suptitle(clf.model_name)
            if col == 0:
                subfig.supylabel(ae.model_name)

    return fig
