from typing import List, Type

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchtyping import TensorType

from tabsplanation.data import NormalizeInverse
from tabsplanation.types import InputPoint

AutoEncoder = Type["AutoEncoder"]


def roundtrip_test(
    x: TensorType["nb_points", "input_dim"], ae: AutoEncoder, nb_roundtrips: int
) -> TensorType["nb_roundtrips", "nb_points", "input_dim"]:
    """Compute up to `nb_roundtrips` roundtrips of the auto-encoder `ae`
    on point `x`.

    Given an input point `x`, apply `decode . encode` to `x`
    `n` times, for `n` from 1 to `nb_roundtrips`.
    """

    def roundtrip(x_: InputPoint) -> InputPoint:
        with torch.no_grad():
            x_reconstructed = ae.decode(ae.encode(x_))
        return x_reconstructed

    x_n = x.clone()

    roundtrips = []
    for n in range(1, nb_roundtrips + 1):
        x_n = roundtrip(x_n)
        roundtrips.append(x_n)

    return torch.stack(roundtrips)


def plot_roundtrip_test(
    inputs: TensorType["nb_points", "input_dim"],
    aes: List[AutoEncoder],
    nb_roundtrips: int,
    normalize_inverse: NormalizeInverse,
) -> mpl.figure.Figure:

    inputs_nonnormalized = normalize_inverse(inputs)

    # The test dataset might be standardized, in which case we'll have
    # to de-standardize at the end.
    distances = []
    for ae in aes:

        roundtrips: TensorType[
            "nb_roundtrips", "nb_points", "input_dim"
        ] = roundtrip_test(inputs, ae, nb_roundtrips)

        roundtrips_nonnormalized = normalize_inverse(roundtrips)

        ae_distances: TensorType["nb_points", "nb_roundtrips"] = torch.stack(
            [
                torch.cdist(
                    inputs_nonnormalized[[i], :], roundtrips_nonnormalized[:, i, :]
                )
                for i in range(len(inputs))
            ]
        ).squeeze()

        distances.append(ae_distances)

    # Pytorch 2D tensors are recognized as 1D array of arrays
    # so we have to convert to numpy
    distances = [np.array(d) for d in distances]

    width, height = plt.rcParams["figure.figsize"]
    width = (width / 3) * len(aes)
    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(aes),
        sharey=True,
        figsize=(width, height),
        layout="tight",
        squeeze=False,
    )
    fig.suptitle(
        r"Distance between $x$ and $(D \circ E)^n(x)$ for $x \in D_\mathrm{test}$"
    )
    fig.supxlabel("Number of roundtrips")
    fig.supylabel(r"$L_2$ distance")

    for i, (ae, ae_errors) in enumerate(zip(aes, distances)):
        ax[0, i].boxplot(ae_errors)
        ax[0, i].set_title(ae.model_name)

    return fig
