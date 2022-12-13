"""Produce counterfactual explanations given a classifier and an
autoencoder.

`cf` stands for "counterfactual".
`prb` stands for "probability".
"""

from typing import Dict, Optional, Union

import numpy as np
import torch

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


class LatentShift:
    def __init__(self, classifier: Classifier, autoencoder: AutoEncoder, hparams: Dict):
        self.classifier = classifier
        self.autoencoder = autoencoder

        self._shift_step = hparams["shift_step"]
        self._max_iter = hparams["max_iter"]

    def get_counterfactuals(
        self,
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
        ae = self.autoencoder
        clf = self.classifier

        # Switch AE to evaluation mode (influences model behaviour in some
        # cases, e.g. `BatchNorm` doesn't compute the mean and standard deviation
        # which means the model can be applied to just one point)
        ae.eval()

        # 1) Make a way to perform latent shift by a given shift

        output = clf.predict_proba(input).detach()
        output_class = torch.argmax(output, dim=-1)
        if target_class == output_class:
            raise ValueError(
                f"The target class is equal to the output class (class {output_class})."
            )

        # Send the input to the latent space
        z = ae.encode(input.reshape((1, -1)))

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
        # max_shift = self._find_max_shift(
        #     ae, clf, z, gradient, output_class, target_class
        # )

        # 2b) Compute latent shifts
        # shifts = np.linspace(-max_shift, max_shift).reshape(-1, 1)
        shifts = torch.tensor(
            [i * self._shift_step for i in range(self._max_iter)]
        ).reshape(-1, 1)
        cf_xs = self._latent_shift(ae, z, gradient, shifts)
        cf_ys = clf.predict_proba(cf_xs)

        # 3) Pack explanations together neatly
        return ExplanationPath(
            explained_input=InputOutputPair(input, output),
            target_class=target_class,
            # maximum_shift=max_shift,
            # shifts=shifts / max_shift,
            shift_step=self._shift_step,
            max_iter=self._max_iter,
            cfs=[InputOutputPair(cf_x, cf_y) for cf_x, cf_y in zip(cf_xs, cf_ys)],
        )

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
        xs = ae.decode(z_perturbed).detach()
        return xs

    def _find_max_shift(
        self,
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
        xs = self._latent_shift(ae, z, gradient, shifts)
        prbs = clf.predict_proba(xs)

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


# Pretty sure this is the same as REVISE
class LatentShiftRecomputeGradient(LatentShift):
    def get_counterfactuals(
        self,
        input: InputPoint,
        target_class: Optional[int],
        clf: Classifier,
        ae: AutoEncoder,
    ) -> ExplanationPath:
        """Produce counterfactual explanations for the given input point.

        As opposed the `make_path`, here we recompute the gradient in latent
        space at each step, to mitigate the effect of the gradient having
        less significance the further away we perturb.

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

        output = clf.predict_proba(input).detach()
        output_class = np.argmax(output)
        if target_class == output_class:
            raise ValueError(
                f"The target class is equal to the output class (class {output_class})."
            )

        # Send the input to the latent space
        z_x = ae.encode(input.reshape((1, -1)))

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
        def get_gradient(z: LatentPoint):
            return torch.autograd.grad(clf_decode(z), z)[0]

        # 2) Apply latent shift, recomputing the gradient at each step

        # Forward (make target class more probable)
        forward_cfs = []
        forward_shifts = []
        shift = 0
        shift_step = 0.001
        criterion = True
        z = z_x
        while criterion:
            z = z + shift * get_gradient(z)
            cf_x = ae.decode(z)
            cf_y = clf.predict_proba(cf_x)

            forward_cfs.append(InputOutputPair(cf_x, cf_y))
            forward_shifts.append(shift)

            criterion = (cf_y.squeeze()[target_class] < 0.5) and shift < 1

            shift = shift + shift_step

        max_shift = shift

        # Backward (make target class less probable)
        backward_cfs = []
        backward_shifts = []
        shift_step = -shift_step
        # Don't start at 0, we already have it
        shift = 0 + shift_step
        nb_forward_shifts = len(forward_shifts)
        z = z_x
        for _ in range(nb_forward_shifts - 1):
            z = z + shift * get_gradient(z)
            cf_x = ae.decode(z)
            cf_y = clf.predict_proba(cf_x)
            backward_cfs.append(InputOutputPair(cf_x, cf_y))
            backward_shifts.append(shift)

            shift = shift + shift_step

        cfs = list(reversed(backward_cfs)) + forward_cfs
        shifts = list(reversed(backward_shifts)) + forward_shifts

        # 3) Pack explanations together neatly
        return ExplanationPath(
            explained_input=InputOutputPair(input, output),
            target_class=target_class,
            maximum_shift=max_shift,
            shifts=torch.tensor(
                [shift / max_shift for shift in shifts], dtype=torch.float
            ),
            cfs=cfs,
        )
