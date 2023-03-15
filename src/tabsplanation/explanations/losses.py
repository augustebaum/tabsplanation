"""Different losses to be used for finding counterfactuals."""
from typing import Literal, Optional

import torch
from torch import nn

from tabsplanation.types import B, C, D, H, RelativeFloat, S, Tensor, V


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
        coef: RelativeFloat = 1.0,
    ):
        super(GeneralValidityLoss, self).__init__()
        self._kind = kind
        self._classes = classes

        if classes == "target":
            self.coef_source = 0.0
            self.coef_others = 0.0
        if classes == "source":
            self.coef_source = coef
            self.coef_others = 0.0
        if classes == "others":
            self.coef_source = coef
            self.coef_others = coef

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
        # logits.gather(2, classes.view(1, -1,1).expand(steps, -1, -1))
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
        # Clamp to avoid infs
        log_prbs = logits.softmax(dim=-1).clamp(min=10 ** (-18)).log()
        return GeneralValidityLoss.get_logits(log_prbs, classes)

    def forward(
        self,
        logits: Tensor[B, C],
        source: Tensor[B, int],
        target: Tensor[B, int],
    ):
        """Compute the loss for a counterfactual whose original point was predicted as
        class `source`, and whose target class is `target`."""

        if logits.dim() == 3:
            # Assume logits has shape S, B, C
            nb_steps = logits.shape[0]
            logits = logits.flatten(end_dim=-2)
            target = target.repeat_interleave(nb_steps)
            source = source.repeat_interleave(nb_steps)

        kind_target: Tensor[B, 1] = self.kind_fn(logits, target)
        kind_source: Tensor[B, 1] = self.kind_fn(logits, source)

        kind_all_classes: Tensor[B, C] = self.kind_fn(logits)
        # Normalize
        kind_others: Tensor[B, 1] = (
            kind_all_classes.sum(dim=1).view(-1, 1) - kind_target - kind_source
        )
        if (
            kind_target.isnan().any()
            or kind_source.isnan().any()
            or kind_others.isnan().any()
        ):
            raise ValueError()

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


class PathLoss(nn.Module):
    def __init__(self):
        super(PathLoss, self).__init__()

    def forward(
        self,
        autoencoder,
        classifier,
        latents: Tensor[B, S, H],
        source_class: Tensor[B],
        target_class: Tensor[B],
    ):
        raise NotImplementedError()


class BoundaryCrossLoss(PathLoss):
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
        prbs: Tensor[B, S, C] = classifier.predict_proba(autoencoder.decode(latents))
        prbs_filtered: Tensor[B, S, 2] = take_source_and_target(
            prbs, source_class, target_class
        )

        prb_max_source_and_target: Tensor[B, S] = prbs_filtered.max(dim=2).values
        return -prb_max_source_and_target.mean()


def where_changes(tensor: Tensor[B, S], dim=-1) -> Tensor[B, S]:
    """Return a Tensor containing 1 if the corresponding entry is
    different from the previous one (according to `dim`), 0 otherwise.

    By convention the result always starts with 0.
    """
    prepend = tensor.index_select(dim, torch.tensor([0]).to(tensor.device))
    diffs = tensor.diff(prepend=prepend, dim=dim)
    result = torch.zeros_like(tensor).to(tensor.device)
    result[diffs != 0] = 1
    return result


def indices_where_changes(tensor: Tensor[S]) -> Tensor[S]:
    """Return a Tensor containing the indices in `tensor` where the entry is
    different from the previous one.

    By convention the result always starts with 0.
    """
    if len(tensor) == 0:
        return tensor
    zero = torch.tensor([0]).to(tensor.device)
    return torch.cat([zero, where_changes(tensor).nonzero().squeeze()])


def get_path_ends(target: Tensor[B], cf_preds: Tensor[S, B]) -> Tensor[V]:
    """Find where the path number changes, which will tell us how many
    steps were taken to reach the target for each step.

    Inputs:
    -------
    * target: The target class, for each point in the batch.
    * cf_preds: The predicted class for each step of each path in the batch.
    """
    path_numbers, step_numbers = torch.where((target == cf_preds).T)
    return step_numbers[indices_where_changes(path_numbers)]


def get_path_mask(cf_preds: Tensor[S, B], target: Tensor[B]) -> Tensor[S, B]:
    """Return a Tensor where each column is 1 until `target` is reached, then 0.
    If the target is not reached then the whole column is 0 (not 1)."""

    # Tells you whether or not
    # `target` was reached in column `j` at or before `i`.
    # `cumsum` is like reducing with a logical "or".
    target_reached_progression: Tensor[S, B] = (target == cf_preds).cumsum(dim=0).bool()
    # Tells you if `target` was reached at any point.
    target_reached: Tensor[B] = target_reached_progression[-1, :]
    # True until target is reached, then False from the
    # first index where target is reached.
    # All False when target is not reached.
    return target_reached_progression.logical_not() * target_reached


class PointLoss(PathLoss):
    """A loss that considers whether the path passed by a given point *before reaching
    the target*.

    It finds the minimum of the distances between the point and all points in the path.
    """

    def __init__(self, point: Tensor[D]):
        super(PointLoss, self).__init__()
        self.register_buffer("point", point)

    def forward(
        self,
        autoencoder,
        classifier,
        latents: Tensor[S, B, H],
        source_class: Tensor[B],
        target_class: Tensor[B],
    ):
        s, b, _ = latents.shape

        inputs: Tensor[S, B, D] = autoencoder.decode(latents)
        inputs_2d: Tensor[S * B, D] = inputs.view(-1, inputs.shape[-1])
        distances_2d: Tensor[S * B] = torch.linalg.vector_norm(
            inputs_2d - self.point, dim=-1
        )
        distances: Tensor[S, B] = distances_2d.view(s, b)
        min_distances: Tensor[B] = distances.amin(dim=0)
        return min_distances.mean()

        # MaskedTensor is too new; it hogs all the memory
        # distances = torch.masked.masked_tensor(
        #     distances_2d.view(s, b), path_mask, requires_grad=True
        # )

        # return distances.amin(dim=0).mean().get_data().nan_to_num(torch.tensor(0))


class MaxPointLoss(PointLoss):
    """A loss that considers whether the path passed by the point
    that has, for each feature, the maximum feature value in the dataset."""

    def __init__(self, dataset: Tensor[B, D]):
        point = dataset.max(dim=0).values.to(dataset.device)
        super(MaxPointLoss, self).__init__(point)
