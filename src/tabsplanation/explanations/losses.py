"""Different losses to be used for finding counterfactuals."""

import torch
from torch import nn

from tabsplanation.types import Tensor


class AwayLoss(nn.Module):
    """Loss computing `CrossEntropy(input, y_source)`."""

    def __init__(self):
        super(AwayLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        input: Tensor["batch_size", "nb_classes"],
        source: Tensor["batch_size", int],
        target: Tensor["batch_size", int] = None,
    ):
        """Compute the loss for a counterfactual whose original point was predicted as
        class `source`, and whose target class is `target`."""
        return -self.ce_loss(input, source)


class BabyStretchLoss(nn.Module):
    """Loss computing `CrossEntropy(input, y_source) - CrossEntropy(input, y_target)`.

    Intuitively, the goal is to bring the input close to `target` and far from
    `source`.
    """

    def __init__(self):
        super(BabyStretchLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        input: Tensor["batch_size", "nb_classes"],
        source: Tensor["batch_size", int],
        target: Tensor["batch_size", int],
    ):
        """Compute the loss for a counterfactual whose original point was predicted as
        class `source`, and whose target class is `target`."""
        return self.ce_loss(input, target) - self.ce_loss(input, source)


class StretchLoss(nn.Module):
    r"""Loss computing
    $-CrossEntropy(input, y_target) + \sum_{c \neq target} {CrossEntropy(input, y_c)}$.

    Intuitively, the goal is to bring the input closer to `target` and far from all the
    other classes.
    """

    def __init__(self):
        super(StretchLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        input: Tensor["batch_size", "nb_classes"],
        target: Tensor["batch_size", int],
        source: Tensor["batch_size", int] = None,
    ):
        """Compute the loss for a counterfactual whose original point was predicted as
        class `source`, and whose target class is `target`."""
        nb_classes = input.shape[-1]

        def class_(class_number):
            return torch.full((len(input),), class_number)

        # This is the same as adding up for all classes except the target,
        # then removing the target once.
        return -sum(
            self.ce_loss(input, class_(class_number))
            for class_number in range(nb_classes)
        ) + 2 * self.ce_loss(input, target)
