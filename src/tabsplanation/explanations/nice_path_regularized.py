from typing import Any, Dict

import torch
from torch import nn

from tabsplanation.models.classifier import Classifier
from tabsplanation.models.normalizing_flow import NICEModel
from tabsplanation.types import Tensor

Explainer = Any


class PathRegularizedNICE(NICEModel):
    def __init__(
        self,
        classifier: Classifier,
        explainer: Explainer,
        autoencoder_args: Dict,
    ):
        super(PathRegularizedNICE, self).__init__(**autoencoder_args)

        self.explainer = explainer
        self.classifier = classifier
        self.nb_classes = self.classifier.layers[-1].out_features

    def step(self, batch, batch_idx):
        # First step: compute likelihood
        loss, logs = super(PathRegularizedNICE, self).step(batch, batch_idx)

        x, y = batch

        latent_paths = self.explainer.get_counterfactuals(
            self.classifier, self, x, self.random_targets_like(y)
        )
        path_loss = self.path_loss_fn(latent_paths)

        logs |= {"path_loss": path_loss}
        loss += path_loss
        return loss, logs

    def random_targets_like(self, y):
        batch_target_difference = torch.randint_like(y, low=1, high=self.nb_classes - 1)
        batch_target_class = (y + batch_target_difference) % self.nb_classes
        return batch_target_class


class ValidityLoss(nn.Module):
    """A loss that considers if the path is valid (i.e. ended up in the target
    class)."""

    def __init__(self, classifier, autoencoder):
        super(ValidityLoss, self).__init__()

        self.autoencoder = autoencoder
        self.classifier = classifier

    def forward(self, prbs: Tensor["nb_steps", "nb_classes"], target: int):
        self.classifier.predict_proba(self.autoencoder.decode(latents))
        prbs
        # path.target


class BoundaryCrossLoss(nn.Module):
    """A loss that considers the time passed in either the source class or the target
    class.

    Intuitively, the path should lie as much as possible either in the source class or
    in the target class.
    """

    def __init__(self):
        super(BoundaryCrossLoss, self).__init__()

    def forward(self, path):
        pass
