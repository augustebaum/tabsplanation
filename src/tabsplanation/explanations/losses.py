"""Different losses to be used for finding counterfactuals."""

import torch
from torch import nn

from tabsplanation.types import B, C, Tensor


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
