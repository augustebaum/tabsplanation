from dataclasses import dataclass
from typing import Annotated, List, Optional, Tuple, TypeAlias, Union

import numpy as np
import torch
from torchtyping import TensorType  # type: ignore

Tensor: TypeAlias = TensorType

InputPoint: TypeAlias = Tensor["input_dim"]
InputBatch: TypeAlias = Tensor["batch", "input_dim"]

OutputPoint: TypeAlias = Tensor["output_dim"]
OutputBatch: TypeAlias = Tensor["batch", "output_dim"]

LatentPoint: TypeAlias = Tensor["latent_dim"]
LatentBatch: TypeAlias = Tensor["batch", "latent_dim"]

Input: TypeAlias = Union[InputPoint, InputBatch]
Output: TypeAlias = Union[OutputPoint, OutputBatch]
Latent: TypeAlias = Union[LatentPoint, LatentBatch]


@dataclass
class ValueBetween:
    """Annotation to specify bounds on a numerical value.

    Attributes:
    -----------
    lo: Lower bound. `None` means there is no lower bound.
    hi: Upper bound. `None` means there is no upper bound.
    exclude_lo: Whether or not `lo` is a valid value.
    exclude_hi: Whether or not `hi` is a valid value.
    """

    lo: Optional[float]
    hi: Optional[float]
    exclude_lo: bool = False
    exclude_hi: bool = False


PositiveFloat: TypeAlias = Annotated[float, ValueBetween(0, None)]
RelativeFloat: TypeAlias = Annotated[float, ValueBetween(0, 1)]

AbsoluteDistance: TypeAlias = PositiveFloat
Distance: TypeAlias = RelativeFloat

Logit: TypeAlias = float
Probability: TypeAlias = RelativeFloat

AbsoluteShift: TypeAlias = float
Shift: TypeAlias = Annotated[float, ValueBetween(-1, 1)]


@dataclass
class InputOutputPair:
    """A pair containing an input point and its predicted probability.

    Attributes:
    -----------
    input: An input point, possibly the result of a perturbation.
    output: The output of a model on the input.
    """

    input: InputPoint
    output: OutputPoint

    @property
    def x(self):
        return self.input.detach()

    @property
    def y(self):
        return self.output.detach()


class LatentShiftPath:
    def __init__(
        self,
        explained_input: InputOutputPair,
        target_class: Optional[int],
        maximum_shift: Tensor["nb_shifts"] | Tensor["nb_shifts", 1],
        shifts: List[Shift],
        cfs: List[InputOutputPair],
    ):
        self.explained_input = explained_input
        self.maximum_shift = maximum_shift
        self.target_class = target_class

        # TensorType["nb_explanations", "input_dim"]
        self.xs = torch.cat([cf.x.unsqueeze(0) for cf in cfs])

        # TensorType["nb_explanations", "input_dim"]
        self.ys = torch.cat([cf.y.unsqueeze(0) for cf in cfs])

        self.shifts = shifts.reshape(-1, 1)

    @property
    def distances(self) -> Tensor:
        """Produce a tensor of relative distances from each explanation
        to the original point."""

        x = self.explained_input.x
        # 1-by-N
        distances = torch.cdist(x.unsqueeze(0), self.xs)
        # distances = distances / max(distances)
        distances = torch.reshape(distances, (-1, 1))
        return distances

    @property
    def original_class(self) -> int:
        return np.argmax(self.explained_input.y)

    @property
    def new_class(self):
        return self.target_class

    @property
    def prb_deltas(self) -> Tensor["nb_explanations", 1]:
        """Returns the difference in probabilities from the explained
        prediction to each counterfactual."""
        cf_prbs = self.prbs_old
        original_prb = self.explained_input.y[self.original_class]
        return cf_prbs - original_prb

    @property
    def prbs_old(self) -> Tensor["nb_explanations", 1]:
        """Returns the probability of the current class for each counterfactual."""
        return self.ys[:, [self.original_class]]

    @property
    def prbs_new(self) -> Tensor["nb_explanations", 1]:
        """Returns the probability of the target class for each counterfactual."""
        if self.target_class is None:
            raise ValueError("The explanation does not have a target class.")
        return self.ys[:, [self.target_class]]

    def as_tuple(self) -> Tuple["ShiftsTensor", "PrbsTensor"]:
        """Returns the relative shifts and the corresponding probabilities
        of the originally predicted class for each counterfactual."""
        return self.shifts, self.prbs_old