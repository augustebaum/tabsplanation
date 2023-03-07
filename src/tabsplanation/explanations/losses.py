"""Different losses to be used for finding counterfactuals."""

from typing import Literal, Optional

import torch
from torch import nn

from tabsplanation.types import B, C, D, H, S, Tensor


class ValidityLoss(nn.Module):
    def __init__(self):
        super(ValidityLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        logits: Tensor[B, C],
        source: Tensor[B, int],
        target: Tensor[B, int],
    ):
        """Compute the loss for a counterfactual whose original point was predicted as
        class `source`, and whose target class is `target`."""
        raise NotImplementedError("This is an abstract class")


class GeneralValidityLoss(nn.Module):
    def __init__(
        self,
        kind: Literal["prb", "log_prb", "logit"],
        classes: Literal["target", "source", "others"],
    ):
        super(GeneralValidityLoss, self).__init__()
        # self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self._kind = kind
        self._classes = classes

        if classes == "target":
            self.coef_source = 0
            self.coef_others = 0
        if classes == "source":
            self.coef_source = 1
            self.coef_others = 0
        if classes == "others":
            self.coef_source = 1
            self.coef_others = 1

        if kind == "logit":
            self.kind_fn = GeneralValidityLoss.get_logits
        elif kind == "prb":
            self.kind_fn = GeneralValidityLoss.get_prbs
        elif kind == "log_prb":
            self.kind_fn = GeneralValidityLoss.get_log_prbs
        else:
            raise NotImplementedError(
                'Kind not recognized. Accepted values are "logit", "prb", "log_prb".'
            )

    def __str__(self):
        return f"{self.__class__.__name__}(kind={self._kind}, classes={self._classes})"

    @staticmethod
    def get_logits(
        logits: Tensor[B, C],
        classes: Optional[Tensor[B, "k"]] = None,
    ) -> Tensor[B, "k"]:
        if classes is None:
            return logits
        if len(classes.shape) == 1:
            return logits.gather(1, classes.view(-1, 1))
        return logits.gather(1, classes)

    @staticmethod
    def get_prbs(
        logits: Tensor[B, C],
        classes: Optional[Tensor[B, "k"]] = None,
    ) -> Tensor[B, "k"]:
        prbs = logits.softmax(dim=-1)
        return GeneralValidityLoss.get_logits(prbs, classes)

    @staticmethod
    def get_log_prbs(
        logits: Tensor[B, C],
        classes: Optional[Tensor[B, "k"]] = None,
    ) -> Tensor[B, "k"]:
        log_prbs = logits.softmax(dim=-1).log()
        return GeneralValidityLoss.get_logits(log_prbs, classes)

    def forward(
        self,
        logits: Tensor[B, C],
        source: Tensor[B, int],
        target: Tensor[B, int],
    ):
        """Compute the loss for a counterfactual whose original point was predicted as
        class `source`, and whose target class is `target`."""

        kind_target: Tensor[B, 1] = self.kind_fn(logits, target)
        kind_source: Tensor[B, 1] = self.kind_fn(logits, source)

        kind_all_classes: Tensor[B, C] = self.kind_fn(logits)
        kind_others: Tensor[B, 1] = (
            kind_all_classes.sum(dim=1).view(-1, 1) - kind_target - kind_source
        )

        return (
            -kind_target
            + self.coef_source * kind_source
            + self.coef_others * kind_others
        )


class LogitValidityLoss(ValidityLoss):
    def __init__(self, reg_target):
        super(ValidityLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.reg_target = reg_target

    def forward(
        self,
        logits: Tensor[B, C],
        source: Tensor[B],
        target: Tensor[B],
    ) -> Tensor[B]:
        r"""Compute $- (reg_target + 1) x_t + \sum_c x_c$."""
        target_logits = logits.gather(1, target.reshape(-1, 1))

        return logits.sum(dim=1) - (1 + self.reg_target) * target_logits.squeeze()


class AwayLoss(ValidityLoss):
    """Loss computing `-CrossEntropy(input, y_source)`."""

    def __init__(self):
        super(AwayLoss, self).__init__()

    def forward(
        self,
        logits: Tensor[B, C],
        source: Tensor[B, int],
        target: Tensor[B, int] = None,
    ):
        """Compute the loss for a counterfactual whose original point was predicted as
        class `source`, and whose target class is `target`."""
        return -self.ce_loss(input, source)


class TargetLoss(ValidityLoss):
    """Loss computing `CrossEntropy(input, y_target)`.

    Intuitively, the goal is to bring the input close to `target`.
    """

    def __init__(self):
        super(TargetLoss, self).__init__()

    def forward(
        self,
        logits: Tensor[B, C],
        source: Tensor[B, int],
        target: Tensor[B, int],
    ):
        """Compute the loss for a counterfactual whose original point was predicted as
        class `source`, and whose target class is `target`."""
        return self.ce_loss(logits, target)


class BinaryStretchLoss(ValidityLoss):
    """Loss computing `CrossEntropy(input, y_target) - CrossEntropy(input, y_source)`.

    Intuitively, the goal is to bring the input close to `target` and far from
    `source`.
    """

    def __init__(self):
        super(BinaryStretchLoss, self).__init__()

    def forward(
        self,
        logits: Tensor[B, C],
        source: Tensor[B, int],
        target: Tensor[B, int],
    ):
        """Compute the loss for a counterfactual whose original point was predicted as
        class `source`, and whose target class is `target`."""
        return (self.ce_loss(logits, target) - self.ce_loss(logits, source)) / 2


class StretchLoss(ValidityLoss):
    r"""Loss computing
    $CrossEntropy(input, y_target) - \sum_{c \neq target} {CrossEntropy(input, y_c)}$.

    Intuitively, the goal is to bring the input closer to `target` and far from all the
    other classes.
    """

    def __init__(self):
        super(StretchLoss, self).__init__()

    def forward(
        self,
        logits: Tensor[B, C],
        source: Tensor[B, int],
        target: Tensor[B, int],
    ):
        """Compute the loss for a counterfactual whose original point was predicted as
        class `source`, and whose target class is `target`."""
        nb_classes = logits.shape[-1]

        def class_(class_number):
            return torch.full((len(logits),), class_number).to(logits.device)

        # This is the same as adding up for all classes except the target,
        # then removing the target once.
        return (
            -sum(
                self.ce_loss(logits, class_(class_number))
                for class_number in range(nb_classes)
            )
            + 2 * self.ce_loss(logits, target)
        ) / nb_classes


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
        latents_2d = latents.view(-1, latents.shape[2])
        inputs = autoencoder.decode(latents_2d)
        logits_2d: Tensor[B * S, C] = classifier(inputs)
        logits: Tensor[B, S, C] = logits_2d.view(
            latents.shape[0], latents.shape[1], logits_2d.shape[1]
        )
        logits_filtered: Tensor[B, S, 2] = take_source_and_target(
            logits, source_class, target_class
        )
        logit_max_source_and_target: Tensor[B, S] = logits_filtered.max(dim=2).values
        return -logit_max_source_and_target.mean()


class PointLoss(nn.Module):
    """A loss that considers whether the path passed by a given point."""

    def __init__(self, point: Tensor[D]):
        super(PointLoss, self).__init__()
        self.point = point

    def forward(
        self,
        autoencoder,
        classifier,
        latents: Tensor[B, S, H],
        source_class: Tensor[B],
        target_class: Tensor[B],
    ):
        latents_2d: Tensor[B * S, H] = latents.view(-1, latents.shape[2])
        inputs_2d: Tensor[B * S, D] = autoencoder.decode(latents_2d)
        distances_2d: Tensor[B * S] = torch.linalg.vector_norm(
            inputs_2d - self.point, dim=-1
        )
        distances: Tensor[B, S] = distances_2d.view(latents.shape[0], latents.shape[1])
        min_distances: Tensor[B] = distances.min(dim=-1).values
        return min_distances.mean()


class MaxPointLoss(PointLoss):
    """A loss that considers whether the path passed by the point
    that has, for each feature, the maximum feature value in the dataset."""

    def __init__(self, dataset: Tensor[B, D]):
        point = dataset.max(dim=0).values
        super(MaxPointLoss, self).__init__(point)
