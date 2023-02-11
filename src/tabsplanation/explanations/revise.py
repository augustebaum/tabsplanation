"""
Heavily adapted port of REVISE algorithm from CARLA repository catalog.
This is because installing the `carla-recourse` package into
this project doesn't work because of dependency and design conflicts.

You can find the original implementation here:
<https://github.com/carla-recourse/CARLA/blob/9595d4f6609ff604bc22d9b8e6cd728ecf18737b/carla/recourse_methods/catalog/revise/model.py>
"""

from typing import Dict, Iterator, List, Optional

import torch

from tabsplanation.explanations.losses import BinaryStretchLoss, ValidityLoss
from tabsplanation.models.autoencoder import AutoEncoder
from tabsplanation.models.classifier import Classifier
from tabsplanation.types import ExplanationPath, InputOutputPair, InputPoint, Tensor


class Revise:
    """
    Implementation of Revise from Joshi et.al. [1]_.

    Parameters
    ----------
    classifier: Classifier model to be explained.
    autoencoder: AutoEncoder (or normalizing flow) model used for producing the
        latent space.
    hparams: Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to
        initialize.
        Please make sure to pass all values as dict with the following keys.

        * "optimizer": {"adam", "rmsprop"}
            Optimizer for generation of counterfactuals.
        * "lr": float
            Learning rate for Revise.
        * "max_iter": int
            Number of iterations for Revise optimization.
        * "distance_regularization": float
            Coefficient for the $L_1$ distance regularization.

    .. [1] Shalmali Joshi, Oluwasanmi Koyejo, Warut Vijitbenjaronk, Been Kim, and Joydeep Ghosh.2019.
            Towards Realistic Individual Recourse and Actionable Explanations in Black-Box Decision Making Systems.
            arXiv preprint arXiv:1907.09615(2019).
    """

    def __init__(
        self,
        classifier: Classifier,
        autoencoder: AutoEncoder,
        hparams: Dict,
        validity_loss: ValidityLoss = BinaryStretchLoss,
    ) -> None:

        self._distance_reg = hparams["distance_regularization"]
        self._optimizer = hparams["optimizer"]
        self._lr = hparams["lr"]
        self._max_iter = hparams["max_iter"]

        self.classifier = classifier
        self.autoencoder = autoencoder

    def get_counterfactuals(
        self,
        input: InputPoint,
        target_class: Optional[int],
    ) -> ExplanationPath:

        classifier = self.classifier
        autoencoder = self.autoencoder

        z = autoencoder.encode(input.reshape(1, -1))
        z = z.clone().detach().requires_grad_(True)
        if self._optimizer == "adam":
            optim = torch.optim.Adam([z], self._lr)
        else:
            optim = torch.optim.RMSprop([z], self._lr)

        paths: List[ExplanationPath] = self._iterate_counterfactual_optimization(
            classifier, autoencoder, optim, z, input, target_class
        )

        return paths[0] if len(paths) == 1 else paths

    def get_counterfactuals_iterator(
        self,
        input: InputPoint,
        target_class: Optional[int],
    ) -> Iterator[ExplanationPath]:

        classifier = self.classifier
        autoencoder = self.autoencoder

        z = autoencoder.encode(input.reshape(1, -1))
        z = z.clone().detach().requires_grad_(True)
        if self._optimizer == "adam":
            optim = torch.optim.Adam([z], self._lr)
        else:
            optim = torch.optim.RMSprop([z], self._lr)

        paths: Iterator[ExplanationPath] = self.get_paths_iterator(
            classifier, autoencoder, optim, z, input, target_class
        )

        return paths

    def _iterate_counterfactual_optimization(
        self, classifier, autoencoder, optim, z, input, target_class
    ):

        paths = self.get_paths_iterator(
            classifier, autoencoder, optim, z, input, target_class
        )
        cf_xs = [path.xs for path in paths]

        return torch.stack(cf_xs)

    def get_paths_iterator(
        self, classifier, autoencoder, optim, z, input, target_class
    ) -> Iterator[ExplanationPath]:

        cf_xs = []
        for _ in range(self._max_iter):

            cf_x = autoencoder.decode(z)

            loss, logs = self._compute_loss(classifier, input, cf_x, target_class)

            loss.backward()
            optim.step()
            optim.zero_grad()
            cf_x.detach_()

            cf_xs.append(cf_x)

        paths_tensor: Tensor["max_iter", "batch", "input_dim"] = torch.stack(cf_xs)

        paths_tensor: Tensor["batch", "max_iter", "input_dim"] = paths_tensor.reshape(
            paths_tensor.shape[1], paths_tensor.shape[0], input.shape[-1]
        )

        return iter(
            ExplanationPath(
                explained_input=InputOutputPair(
                    input=input, output=self.classifier.predict_proba(input)
                ),
                target_class=target_class,
                shift_step=None,
                max_iter=self._max_iter,
                xs=path,
                ys=self.classifier.predict_proba(path),
            )
            for input, target_class, path in zip(input, target_class, paths_tensor)
        )

    def _compute_loss(self, classifier, original_input, cf, target_class):

        y_predict = classifier.predict_proba(original_input)
        source_class = classifier.predict(original_input)

        validity_loss = self.validity_loss(y_predict, source_class, target_class)
        distance_loss = torch.norm((original_input - cf), 1)

        loss = validity_loss + self._distance_reg * distance_loss
        logs = {
            "loss": loss,
            "validity_loss": validity_loss,
            "distance_loss": distance_loss,
        }

        return loss, logs


class ReviseNoDescent(Revise):
    """Like Revise, but without recomputing the gradient at each step (only
    compute it once at the beginning)."""

    def __init__(self, classifier, autoencoder, hparams):
        super(ReviseNoDescent, self).__init__(classifier, autoencoder, hparams)

    def _iterate_counterfactual_optimization(self, optim, z, input, target_class):

        cfs = []

        cf_x = self.autoencoder.decode(z)
        cfs.append(InputOutputPair(cf_x, self.classifier.predict_proba(cf_x)))

        loss, logs = self._compute_loss(input, cf_x, target_class)

        optim.zero_grad()
        loss.backward()
        optim.step()
        cf_x.detach_()

        for _ in range(self._max_iter - 1):

            cf_x = self.autoencoder.decode(z)
            prbs = self.classifier.predict_proba(cf_x).squeeze()
            cfs.append(InputOutputPair(cf_x, prbs))

            loss, logs = self._compute_loss(input, cf_x, target_class)

            if prbs[target_class] > 0.5:
                break

            optim.step()
            cf_x.detach_()

        return cfs
