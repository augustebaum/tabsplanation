"""Produce counterfactual explanations given a classifier and an
autoencoder.

`cf` stands for "counterfactual".
`prb` stands for "probability".
"""

from typing import Optional, Union

import numpy as np
import torch
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from torchtyping import TensorType

from .autoencoder.architectures import AutoEncoder
from .classifier import Classifier
from .types import (
    AbsoluteShift,
    Input,
    InputOutputPair,
    InputPoint,
    LatentPoint,
    LatentShiftPath,
    Logit,
)


def latent_shift(
    ae: AutoEncoder,
    z: LatentPoint,
    gradient: LatentPoint,
    shift: Union[AbsoluteShift, TensorType["nb_shifts", 1]],
) -> Input:
    if not isinstance(shift, torch.Tensor):
        shift = torch.tensor(shift)
    z_perturbed = (z + shift * gradient).to(torch.float)
    xs = ae.decode(z_perturbed).detach()
    return xs


def find_max_shift(
    ae: AutoEncoder,
    clf: Classifier,
    z: LatentPoint,
    gradient: LatentPoint,
    output_class: int,
    target_class: int,
) -> AbsoluteShift:
    """Find maximum shift.

    If we have a target class, then keep searching until
    the target class is predicted with >50% confidence.

    Otherwise, then keep searching until
    some class is predicted with >50% confidence
    (that is not the current class).
    """

    # Compute probabilities for many shifts
    # Increasing sequence of shifts
    shifts = np.logspace(-3, 1).reshape(-1, 1)
    xs = latent_shift(ae, z, gradient, shifts)
    prbs = clf.softmax(xs)

    # Take out the probabilities that don't interest us
    if target_class is None:
        nb_classes = prbs.shape[1]
        column_selector = list(set(range(nb_classes)) - {output_class})
        prbs = prbs[:, column_selector]
    else:
        prbs = prbs[:, [target_class]]

    # Find the first shift for which the probability is
    # over the threshold
    # If there is no such probability, try again with
    # a smaller threshold
    indices = []
    threshold = 0.5
    while len(indices) == 0:
        indices = (prbs > threshold).nonzero()[:, 0]
        threshold -= 0.05

    # If the length is not 0, it's either that some probability is
    # more than threshold, or that the threshold is negative or zero,
    # in which case we should just take the maximum possible shift

    # We assume the list of shifts is increasing
    if threshold <= 0:
        max_shift_idx = -1
    else:
        max_shift_idx = indices[0]

    max_shift = shifts[max_shift_idx].item()
    return max_shift


def make_path(
    input: InputPoint,
    target_class: Optional[int],
    clf: Classifier,
    ae: AutoEncoder,
) -> LatentShiftPath:
    """Produce counterfactual explanations for the given input point.

    The class of the explanation should be different from the originally
    predicted class, and one can optionally specify the desired target
    class.

    Adapted from:
    <https://github.com/mlmed/gifsplanation/blob/fe04d5ce149102289d3df31b1e41f08ab6ce33ee/attribution.py#L18>

    Inputs:
    -------
    input: Base input point to be perturbed. The point should be consistent
        with the pre-processing used to train the auto-encoder and classifier.
    target_class: Optional desired target class of the counterfactual. Should
        be different from the originally predicted class.
    clf: Classifier model to be explained.
    ae: Auto-encoder model used to learn the latent representation. It should
        implement an `encode` and a `decode` method.

    Returns:
    --------
    explanations: Iterable of tuples, each of which containing a perturbed
        input point along with its associated prediction probability.
    """

    # Switch AE to evaluation mode (influences model behaviour in some
    # cases, e.g. `BatchNorm` doesn't compute the mean and standard deviation
    # which means the model can be applied to just one point)
    ae.eval()

    # 1) Make a way to perform latent shift by a given shift

    output = clf.softmax(input).detach()
    output_class = np.argmax(output)
    if target_class == output_class:
        raise ValueError(
            f"The target class is equal to the output class (class {output_class})."
        )

    # Send the input to the latent space
    z = ae.encode(input)

    # Set up the gradient computation
    input.requires_grad = True

    # This is the function of which we take the gradient.
    # The gradient is the direction of steepest ascent.
    # We expect that when we go in the direction of the gradient,
    # the probability of the current class decreases.
    def clf_decode(z: LatentPoint) -> Logit:
        ps = clf(ae.decode(z)).squeeze()
        if target_class is None:
            # The probability of the current class should _decrease_
            # as the shift increases
            return -ps[output_class]
        else:
            # The probability of the current class should _decrease_
            # and the probability of the target class should _increase_
            # as the shift increases
            return ps[target_class] - ps[output_class]

    # Compute gradient of classifier at `z` in latent space
    gradient = torch.autograd.grad(clf_decode(z), z)[0]

    # 2) Apply latent shift

    # 2a) Figure out how far we should shift
    max_shift = find_max_shift(ae, clf, z, gradient, output_class, target_class)

    # 2b) Compute latent shifts
    shifts = np.linspace(-max_shift, max_shift).reshape(-1, 1)
    cf_xs = latent_shift(ae, z, gradient, shifts)
    cf_ys = clf.softmax(cf_xs)

    # 3) Pack explanations together neatly
    return LatentShiftPath(
        explained_input=InputOutputPair(input, output),
        target_class=target_class,
        maximum_shift=max_shift,
        shifts=shifts / max_shift,
        cfs=[InputOutputPair(cf_x, cf_y) for cf_x, cf_y in zip(cf_xs, cf_ys)],
    )


def add_legend(fig, handles):
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        # loc=9,
        fancybox=True,
        shadow=True,
    )


# TODO: Annotate
def make_line_collection(shifts, values, color, label):
    """Make a `LineCollection` from some shifts and the corresponding
    values."""
    n = len(values)
    segments = [torch.cat((s, v), dim=1) for s, v in zip(shifts, values)]
    colors = [mcolors.to_rgba(color, alpha=0.2)] * n
    line_widths = [2] * n

    line_collection = LineCollection(
        segments=segments, colors=colors, linewidths=line_widths, label=label
    )
    return line_collection


# TODO: Annotate
def make_explanations_measurements(explanations):
    """For each input in `inputs`, generate some explanations and record the
    shifts and other information."""

    distances = []
    probabilities = []
    shifts = []

    for explanation in explanations:

        shifts_one, probabilities_one = explanation.as_tuple()

        distances.append(explanation.distances)
        shifts.append(shifts_one)
        probabilities.append(probabilities_one)

    return shifts, probabilities, distances
