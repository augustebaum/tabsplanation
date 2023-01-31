"""Produce counterfactual explanations given a classifier and an
autoencoder.

`cf` stands for "counterfactual".
`prb` stands for "probability".
"""

from typing import Dict, Optional, Union

import numpy as np
import torch

from tabsplanation.explanations.losses import AwayLoss, BabyStretchLoss
from tabsplanation.models.autoencoder import AutoEncoder
from tabsplanation.models.classifier import Classifier
from tabsplanation.types import (
    AbsoluteShift,
    ExplanationPath,
    Input,
    InputOutputPair,
    InputPoint,
    LatentPoint,
    Logit,
    Tensor,
)


def take_(index: Tensor["rows"], src: Tensor["rows", "cols"]):
    """For each `i` in `index`, take the `i`th element of that row in `src`.

    Example:
    --------
    >>> src = torch.arange(1, 11).reshape((5, 2))
    tensor([[ 1,  2],
            [ 3,  4],
            [ 5,  6],
            [ 7,  8],
            [ 9, 10]])
    >>> index = torch.tensor([0, 1, 1, 0, 1])
    >>> take_(index, src)
    tensor([[ 1],
            [ 4],
            [ 6],
            [ 7],
            [10]])
    """
    return torch.gather(src, -1, index.reshape((-1, 1)))


def grad(losses: Tensor["rows", 1], inputs: Tensor["rows", "cols"]):
    """Compute gradient of `losses` with respect to `inputs`.

    The loss at row `i` should correspond to the input at row `i`.
    """
    return torch.autograd.grad(losses.sum(), inputs, retain_graph=True)[0]


class LatentShiftNew:
    def __init__(self, hparams: Dict):
        # self.classifier = classifier
        # self.autoencoder = autoencoder

        self._shift_step = hparams["shift_step"]
        self._max_iter = hparams["max_iter"]

    def get_counterfactuals(
        self,
        classifier: Classifier,
        autoencoder: AutoEncoder,
        input: InputPoint,
        target_class: Optional[int],
    ) -> ExplanationPath:
        """Produce counterfactual explanations for the given input point.

        The class of the explanation should be different from the originally
        predicted class, and one can optionally specify the desired target
        class.

        Adapted from:
        <https://github.com/mlmed/gifsplanation/blob/fe04d5ce149102289d3df31b1e41f08ab6ce33ee/attribution.py#L18>

        Inputs:
        -------
        * classifier: Classifier model to be explained.
        * autoencoder: Auto-encoder model used to learn the latent representation.
            It should implement an `encode` and a `decode` method.
        * input: Base input point to be perturbed. The point should be consistent
            with the pre-processing used to train the auto-encoder and classifier.
        * target_class: Optional desired target class of the counterfactual. It should
            be different from the originally predicted class.

        Returns:
        --------
        explanations: Iterable of tuples, each of which containing a perturbed
            input point along with its associated prediction probability.
        """
        latent_paths = self.get_cf_latents(classifier, autoencoder, input, target_class)
        return latent_paths

    def _latent_shift(
        self,
        ae: AutoEncoder,
        z: LatentPoint,
        gradient: LatentPoint,
        shift: Union[AbsoluteShift, Tensor["nb_shifts", 1]],
    ) -> Input:
        if not isinstance(shift, torch.Tensor):
            shift = torch.tensor(shift)
        z_perturbed = (z + shift * gradient).to(torch.float)
        xs: Tensor["batch", "nb_shifts", "input_dim"] = torch.stack(
            [ae.decode(z_perturbed[:, i, :]) for i in range(z.shape[0])]
        )
        return xs

    def _filter_path(self, cf_xs, cf_ys, output_class, target_class):
        """Cut the path as soon as the target class is reached
        (or some other class is reached, if the target class
        is not specified).
        """
        prbs = cf_ys.clone()

        # Take out the probabilities that don't interest us
        if target_class is None:
            nb_classes = prbs.shape[1]
            column_selector = list(set(range(nb_classes)) - {output_class})
            prbs = prbs[:, column_selector]
        else:
            prbs = prbs[:, [target_class]]

        threshold = 0.5
        idx_above_threshold = (prbs > threshold).nonzero()[:, 0]

        if len(idx_above_threshold) == 0:
            return cf_xs, cf_ys
        first_idx_above_threshold = idx_above_threshold[0]
        return (
            cf_xs[: first_idx_above_threshold + 1],
            cf_ys[: first_idx_above_threshold + 1],
        )

    def get_cf_latents(
        self,
        classifier: Classifier,
        autoencoder: AutoEncoder,
        input: InputPoint,
        target_class: Optional[int],
    ):

        ae = autoencoder
        clf = classifier

        # Switch AE to evaluation mode (influences model behaviour in some
        # cases, e.g. `BatchNorm` doesn't compute the mean and standard deviation
        # which means the model can be applied to just one point)
        # ae.eval()

        # 1) Make a way to perform latent shift by a given shift

        output = clf.predict_proba(input).detach()
        output_class = torch.argmax(output, dim=-1)
        if any(target_class == output_class):
            raise ValueError(
                f"The target class is equal to the output class for at least one point."
            )

        # This is the function of which we take the gradient.
        # The gradient is the direction of steepest ascent.
        # We expect that when we go in the direction of the gradient,
        # the probability of the current class decreases.
            if target_class is None:
            validity_loss_fn = AwayLoss()
            else:
            validity_loss_fn = BabyStretchLoss()

        def clf_decode(z: LatentPoint):
            logits = clf(ae.decode(z))
            return validity_loss_fn(logits, source_class, target_class)

        # Set up the gradient computation
        input.requires_grad = True

        # Send the input to the latent space
        with torch.enable_grad():
            z = ae.encode(input)

            # Compute gradient of classifier at `z` in latent space
            # Multiply by -1 to minimize the loss
            gradient: Tensor["batch", "latent_dim"] = -grad(clf_decode(z), z)

        # 2) Apply latent shift

        shifts = (
            torch.tensor([i * self._shift_step for i in range(self._max_iter)])
            .reshape(-1, 1, 1)
            .to(ae.device)
        )
        z_perturbed = (z + shifts * gradient).to(torch.float)

        return z_perturbed
