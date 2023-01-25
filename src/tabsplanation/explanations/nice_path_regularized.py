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

        self.path_loss_fn = BoundaryCrossLoss()

    def step(self, batch, batch_idx):
        # First step: compute likelihood
        loss, logs = super(PathRegularizedNICE, self).step(batch, batch_idx)

        x, y_source = batch
        y_target: Tensor["batch"] = self.random_targets_like(y_source)

        latent_paths = self.explainer.get_counterfactuals(
            self.classifier, self, x, y_target
        )
        path_loss = self.path_loss_fn(
            self, self.classifier, latent_paths, y_source, y_target
        )

        logs |= {"path_loss": path_loss}
        loss += path_loss
        return loss, logs

    def random_targets_like(self, y):
        batch_target_difference = torch.randint_like(y, low=1, high=self.nb_classes - 1)
        batch_target_class = (y + batch_target_difference) % self.nb_classes
        return batch_target_class


# class ValidityLoss(nn.Module):
#     """A loss that considers if the path is valid (i.e. ended up in the target
#     class)."""

#     def __init__(self, classifier, autoencoder):
#         super(ValidityLoss, self).__init__()

#     def forward(self, latents: Tensor["batch", "nb_steps", "latent_dim"], target: int):
#         latents_2d = latents.reshape(-1, latents.shape[2])
#         inputs = self.autoencoder.decode(latents_2d)
#         prbs = self.classifier.predict_proba(inputs)
#         self.classifier.predict_proba(self.autoencoder.decode(latents))
#         prbs
#         # path.target


class BoundaryCrossLoss(nn.Module):
    """A loss that considers the time passed in either the source class or the target
    class.

    Intuitively, the path should lie as much as possible either in the source class or
    in the target class.
    """

    def __init__(self):
        super(BoundaryCrossLoss, self).__init__()

    def forward(
        self,
        autoencoder,
        classifier,
        latents: Tensor["batch", "nb_steps", "latent_dim"],
        source_class: Tensor["batch"],
        target_class: Tensor["batch"],
    ):
        latents_2d = latents.reshape(-1, latents.shape[2])
        inputs = autoencoder.decode(latents_2d)
        prbs: Tensor["batch * nb_steps", "nb_classes"] = classifier.predict_proba(
            inputs
        )
        prbs: Tensor["batch", "nb_steps", "nb_classes"] = prbs.reshape(
            latents.shape[0], latents.shape[1], prbs.shape[1]
        )
        import pdb

        pdb.set_trace()
        prbs_filtered: Tensor["batch", "nb_steps", 2] = prbs.gather()
        return None

    def take_source_and_target(
        input: Tensor["batch", "nb_steps", "nb_classes"],
        source_class: Tensor["batch"],
        target_class: Tensor["batch"],
    ):
        """
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
                xx[i][:, [src, tgt]]
                for i, src, tgt in enumerate(zip(source_class, target_class))
            ]
        )
