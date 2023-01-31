"""Produce counterfactual explanations given a classifier and an autoencoder.

`cf` stands for "counterfactual".
`prb` stands for "probability".
"""

from typing import Dict, List, Optional

import torch

from tabsplanation.explanations.losses import AwayLoss, BabyStretchLoss
from tabsplanation.models.autoencoder import AutoEncoder
from tabsplanation.models.classifier import Classifier
from tabsplanation.types import (
    ExplanationPath,
    InputOutputPair,
    InputPoint,
    LatentPoint,
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
    tensor([[ 1],   # Element 0 of row 0
            [ 4],   # Element 1 of row 1
            [ 6],   # Element 1 of row 2
            [ 7],   # Element 0 of row 3
            [10]])  # Element 1 of row 4
    """
    return torch.gather(src, -1, index.reshape((-1, 1)))


def grad(losses: Tensor["rows", 1], inputs: Tensor["rows", "cols"]):
    """Compute gradient of `losses` with respect to `inputs`.

    The loss at row `i` should correspond to the input at row `i`.
    """
    return torch.autograd.grad(losses.sum(), inputs, retain_graph=True)[0]


class LatentShiftNew:
    def __init__(self, classifier, autoencoder, hparams: Dict):
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
        latent_paths: Tensor["nb_shifts", "batch", "latent_dim"] = self.get_cf_latents(
            self.classifier, self.autoencoder, input, target_class
        )
        paths: List[ExplanationPath] = self.get_paths(latent_paths, input, target_class)
        return paths[0] if len(paths) == 1 else paths

    def get_cf_latents(
        self,
        classifier: Classifier,
        autoencoder: AutoEncoder,
        input: Tensor["batch", "input_dim"],
        target_class: Tensor["batch", 1],
    ) -> Tensor["nb_shifts", "batch", "latent_dim"]:

        ae = autoencoder
        clf = classifier

        # Switch AE to evaluation mode (influences model behaviour in some
        # cases, e.g. `BatchNorm` doesn't compute the mean and standard deviation
        # which means the model can be applied to just one point)
        ae.eval()

        # 1) Make a way to perform latent shift by a given shift

        output = clf.predict_proba(input).detach()
        source_class = torch.argmax(output, dim=-1)
        if any(target_class == source_class):
            raise ValueError(
                "The target class is equal to the source class for at least one point."
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
            z: Tensor["batch", "latent_dim"] = ae.encode(input)

            # Compute gradient of classifier at `z` in latent space
            # Multiply by -1 to minimize the loss
            gradient: Tensor["batch", "latent_dim"] = -grad(clf_decode(z), z)

        # 2) Apply latent shift

        shifts: Tensor["nb_shifts", 1, 1] = (
            torch.tensor([i * self._shift_step for i in range(self._max_iter)])
            .reshape(-1, 1, 1)
            .to(ae.device)
        )
        z_perturbed: Tensor["nb_shifts", "batch", "latent_dim"] = (
            z + shifts * gradient
        ).to(torch.float)

        return z_perturbed

    def get_paths(
        self,
        latent_paths: Tensor["nb_shifts", "batch", "latent_dim"],
        input,
        target_class,
    ) -> None:
        latent_dim = latent_paths.shape[-1]
        nb_shifts, batch, latent_dim = latent_paths.shape

        paths_tensor: Tensor[
            "nb_shifts * batch", "input_dim"
        ] = self.autoencoder.decode(latent_paths.reshape(-1, latent_dim))

        paths_tensor: Tensor["batch", "nb_shifts", "input_dim"] = paths_tensor.reshape(
            batch, nb_shifts, input.shape[-1]
        )

        return [
            ExplanationPath(
                explained_input=InputOutputPair(
                    input=input, output=self.classifier.predict_proba(input)
                ),
                target_class=target_class,
                shift_step=self._shift_step,
                max_iter=self._max_iter,
                xs=path,
                ys=self.classifier.predict_proba(path),
            )
            for input, target_class, path in zip(input, target_class, paths_tensor)
        ]
