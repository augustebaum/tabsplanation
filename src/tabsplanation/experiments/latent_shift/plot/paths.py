"""Display explanation paths on a 2-D map."""

from typing import Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from torchtyping import TensorType


def points_to_xy(
    xs: TensorType["n", 2]
) -> Tuple[TensorType["n", 1], TensorType["n", 1]]:
    x = xs[:, 0]
    y = xs[:, 1]
    return x, y


def show_path(
    original_x: TensorType[1, 2],
    cf_xs: TensorType["nb_counterfactuals", 2],
    link_idx: int,
) -> Figure:
    """Display one path on the map.

    `link_idx` denotes the index of the row of `cf_xs` which corresponds to
    `original_x`.
    """
    # Create canvas
    fig, ax = plt.subplots()
    plt.axis([0, 50, 0, 50])

    img = mpimg.imread("data_map.jpg")
    ax.imshow(img, origin="upper", extent=[0, 50, 0, 50])

    # Display the point
    x, y = points_to_xy(original_x)
    ax.scatter(x, y)

    # Display the counterfactuals
    x, y = points_to_xy(cf_xs)
    ax.scatter(x, y)

    # Display the link between `original_x` and `latent_shift(original_x, 0)`
    # "dotted"
    if 0 <= link_idx <= cf_xs.shape[0]:
        x_no_shift = cf_xs[link_idx, :].reshape(1, 2)
        x, y = points_to_xy(torch.cat([original_x, x_no_shift], dim=0))
        plt.plot(x, y, linestyle="dotted", zorder=0)

    return fig


if __name__ == "__main__":
    original_x = torch.Tensor([[20, 30]])
    cf_xs = torch.Tensor(
        [
            [25, 40],
            [25, 37],
            [25, 35],
            [25, 32],
            [25, 28],
            [25, 24],
        ]
    )

    show_path(original_x, cf_xs, 5)
    plt.show()
