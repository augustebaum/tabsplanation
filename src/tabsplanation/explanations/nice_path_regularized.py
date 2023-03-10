from typing import Any, Dict

import torch

from tabsplanation.explanations.losses import BoundaryCrossLoss
from tabsplanation.models.classifier import Classifier
from tabsplanation.models.normalizing_flow import NICEModel
from tabsplanation.types import B, Tensor

Explainer = Any


def random_targets_like(y: Tensor, nb_classes: int):
    batch_target_difference = torch.randint_like(y, low=1, high=nb_classes)
    batch_target_class = (y + batch_target_difference) % nb_classes
    return batch_target_class


class PathRegularizedNICE(NICEModel):
    """NICE, but regularized with some other loss as well as NLL."""

    def __init__(
        self,
        classifier: Classifier,
        explainer_cls: Explainer,
        explainer_hparams: Dict,
        autoencoder_args: Dict,
        hparams: Dict,
        path_loss_fn=BoundaryCrossLoss(),
    ):
        super(PathRegularizedNICE, self).__init__(**autoencoder_args)

        self.explainer_cls = explainer_cls
        self.explainer_hparams = explainer_hparams
        self.classifier = classifier

        self.nb_classes = self.classifier.layers[-1].out_features

        self.path_loss_fn = path_loss_fn
        self.path_loss_reg = hparams["path_loss_regularization"]

    def explain(self, x, y_target):
        explainer = self.explainer_cls(self.classifier, self, self.explainer_hparams)
        return explainer.get_cf_latents(x, y_target)

    def step(self, batch, batch_idx):
        # First step: compute likelihood
        nll, logs = super(PathRegularizedNICE, self).step(batch, batch_idx)

        x, _ = batch
        y_source: Tensor[B] = self.classifier.predict(x)
        y_target: Tensor[B] = self.random_targets_like(y_source)

        latent_paths = self.explain(x, y_target)
        path_loss = self.path_loss_fn(
            self, self.classifier, latent_paths, y_source, y_target
        )

        logits = self.classifier(self.decode(latent_paths))
        cf_loss = self.cf_loss_fn(logits, y_source, y_target)

        loss = nll + self.path_loss_reg * path_loss + self.cf_loss_reg * cf_loss
        logs |= {
            "max_memory_gb": torch.cuda.max_memory_allocated(self.device) / (1024 ** 3),
            "cf_loss": cf_loss.detach().item(),
            "path_loss": path_loss.detach().item(),
            "loss": loss.detach().item(),
        }
        if loss.isnan():
            raise ValueError("Loss is NaN")
        return loss, logs

    def random_targets_like(self, y):
        return random_targets_like(y, self.nb_classes)
