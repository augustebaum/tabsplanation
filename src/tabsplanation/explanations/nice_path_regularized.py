from typing import Any, Dict

import torch
from torch import nn

from tabsplanation.models.classifier import Classifier
from tabsplanation.models.normalizing_flow import NICEModel
from tabsplanation.types import B, C, H, S, Tensor

Explainer = Any


def random_targets_like(y: Tensor, nb_classes: int):
    batch_target_difference = torch.randint_like(y, low=1, high=nb_classes)
    batch_target_class = (y + batch_target_difference) % nb_classes
    return batch_target_class


class PathRegularizedNICE(NICEModel):
    def __init__(
        self,
        classifier: Classifier,
        explainer_cls: Explainer,
        explainer_hparams: Dict,
        autoencoder_args: Dict,
    ):
        super(PathRegularizedNICE, self).__init__(**autoencoder_args)

        self.explainer_cls = explainer_cls
        self.explainer_hparams = explainer_hparams
        self.classifier = classifier

        self.nb_classes = self.classifier.layers[-1].out_features

        self.path_loss_fn = BoundaryCrossLoss()

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

        loss = nll + path_loss
        logs = {"nll": nll, "path_loss": path_loss, "loss": loss}
        return loss, logs

    def random_targets_like(self, y):
        return random_targets_like(y, self.nb_classes)


class BoundaryCrossLoss(nn.Module):
    """A loss that considers the time passed in either the source class or the target
    class.

    Intuitively, the path should lie as much as possible either in the source class or
    in the target class.
    The loss computes the average probability that the path lies in neither of them (so
    that the goal is indeed to minimize that probability)

    For each `z`, we compute $1 - max{f(Dec(z))_src, f(Dec(z))_tgt}$.
    """

    def __init__(self):
        super(BoundaryCrossLoss, self).__init__()

    def forward(
        self,
        autoencoder,
        classifier,
        latents: Tensor[B, S, H],
        source_class: Tensor[B],
        target_class: Tensor[B],
    ):
        latents_2d = latents.reshape(-1, latents.shape[2])
        inputs = autoencoder.decode(latents_2d)
        prbs: Tensor[B * S, C] = classifier.predict_proba(inputs)
        prbs: Tensor[B, S, C] = prbs.reshape(
            latents.shape[0], latents.shape[1], prbs.shape[1]
        )
        prbs_filtered: Tensor[B, S, 2] = take_source_and_target(
            prbs, source_class, target_class
        )
        prb_source_plus_target: Tensor[B, S] = prbs_filtered.max(dim=2).values
        return 1 - prb_source_plus_target.mean()


def take_source_and_target(
    input: Tensor[B, S, C],
    source_class: Tensor[B],
    target_class: Tensor[B],
) -> Tensor[B, S, 2]:
    """
    Example:
    --------
    >>> input = torch.arange(1, 19).reshape(2,3,3)
    tensor([[[ 1,  2,  3],
             [ 4,  5,  6],
             [ 7,  8,  9]],

            [[10, 11, 12],
             [13, 14, 15],
             [16, 17, 18]]])
    >>> source = torch.tensor([0, 0])
    >>> target = torch.tensor([2, 1])
    >>> take_source_and_target(input, source, target)
    tensor([[[ 1,  3],
             [ 4,  6],
             [ 7,  9]],

            [[10, 11],
             [13, 14],
             [16, 17]]])
    """
    return torch.stack(
        [
            input[:, i, [src, tgt]]
            for i, (src, tgt) in enumerate(zip(source_class, target_class))
        ]
    )
