"""
Heavily adapted port of REVISE algorithm from CARLA repository catalog.
This is because installing the `carla-recourse` package into
this project doesn't work because of dependency and design conflicts.

You can find the original implementation here:
<https://github.com/carla-recourse/CARLA/blob/9595d4f6609ff604bc22d9b8e6cd728ecf18737b/carla/recourse_methods/catalog/revise/model.py>
"""

from typing import Dict, Optional

import torch
from torch import nn

from tabsplanation.models.autoencoder import AutoEncoder
from tabsplanation.models.classifier import Classifier
from tabsplanation.types import ExplanationPath, InputOutputPair, InputPoint


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

    .. [1] Shalmali Joshi, Oluwasanmi Koyejo, Warut Vijitbenjaronk, Been Kim, and Joydeep Ghosh.2019.
            Towards Realistic Individual Recourse and Actionable Explanations in Black-Box Decision Making Systems.
            arXiv preprint arXiv:1907.09615(2019).
    """

    def __init__(
        self, classifier: Classifier, autoencoder: AutoEncoder, hparams: Dict
    ) -> None:

        self._distance_reg = hparams["distance_regularization"]
        self._optimizer = hparams["optimizer"]
        self._lr = hparams["lr"]
        self._max_iter = hparams["max_iter"]

        self.classifier = classifier
        self.autoencoder = autoencoder

    # def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
    def get_counterfactuals(
        self,
        input: InputPoint,
        target_class: Optional[int],
    ) -> ExplanationPath:

        cfs = self._counterfactual_optimization(input, target_class)
        return cfs

    def _counterfactual_optimization(self, input, target_class):

        cfs = []
        z = self.autoencoder.encode(input.reshape(1, -1))
        z = z.clone().detach().requires_grad_(True)

        if self._optimizer == "adam":
            optim = torch.optim.Adam([z], self._lr)
        else:
            optim = torch.optim.RMSprop([z], self._lr)

        for _ in range(self._max_iter):

            cf_x = self.autoencoder.decode(z)
            cfs.append(InputOutputPair(cf_x, self.classifier.predict_proba(cf_x)))

            loss, logs = self._compute_loss(input, cf_x, target_class)

            loss.backward()
            optim.step()
            optim.zero_grad()
            cf_x.detach_()

        return ExplanationPath(
            explained_input=InputOutputPair(
                input, self.classifier.predict_proba(input)
            ),
            target_class=target_class,
            shift_step=None,
            max_iter=self._max_iter,
            cfs=cfs,
        )

    def _compute_loss(self, original_input, cf, target_class):

        # For multi-class tasks
        loss_function = nn.CrossEntropyLoss()
        output = self.classifier.predict_proba(original_input)

        classification_loss = loss_function(output, target_class)
        distance_loss = torch.norm((original_input - cf), 1)

        loss = classification_loss + self._distance_reg * distance_loss
        logs = {
            "loss": loss,
            "classification_loss": classification_loss,
            "distance_loss": distance_loss,
        }

        return loss, logs
