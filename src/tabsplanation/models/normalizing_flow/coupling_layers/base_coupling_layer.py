import torch
from torch import nn


class BaseCouplingLayer(nn.Module):
    """An implementation of the coupling layer architecture presented in the NICE
    paper."""

    def __init__(self, partition, non_linear_model):
        """Initialize.

        Inputs:
        -------
        * partition: "even" or "odd", denotes the columns that will be passed through
            the non activation.
        * non_linear_model: a function (typically a `nn.Module`) to be applied to a
            partition of the columns. Models the $m$ function from the NICE paper.
        """
        super(BaseCouplingLayer, self).__init__()
        self.partition = partition

        # Make a shortcut for partitions. `first` denotes the columns that will go
        # through the non-linear transformation.
        if partition == "even":
            self.first = _get_even_columns
            self.second = _get_odd_columns
        else:
            self.first = _get_odd_columns
            self.second = _get_even_columns

        # Add the `m` function of the coupling layer. This function is to be learned.
        self.add_module("non_linear_model", non_linear_model)

    def coupling_law(self, a, b):
        raise NotImplementedError

    def anticoupling_law(self, a, b):
        raise NotImplementedError

    def forward(self, x):
        return _interleave(
            self.first(x),
            self.coupling_law(self.second(x), self.non_linear_model(self.first(x))),
            self.partition,
        )

    def inverse(self, z):
        return _interleave(
            self.first(z),
            self.anticoupling_law(self.second(z), self.non_linear_model(self.first(z))),
            self.partition,
        )


def _get_even_columns(x):
    return x[:, 0::2]


def _get_odd_columns(x):
    return x[:, 1::2]


def _interleave(first, second, order):
    """Given two rank-2 tensors with the same batch dimension, interleave their columns.

    * first: Tensor[B, M].
    * second: Tensor[B, N] with $N = M$ or $N = M-1$.
    * order: "even" or "odd", whether the first column of the result should be the
        first column of `first` or that of `second`, respectively.
    """
    cols = []
    if order == "even":
        for k in range(second.shape[1]):
            cols.append(first[:, k])
            cols.append(second[:, k])
        if first.shape[1] > second.shape[1]:
            cols.append(first[:, -1])
    else:
        for k in range(first.shape[1]):
            cols.append(second[:, k])
            cols.append(first[:, k])
        if second.shape[1] > first.shape[1]:
            cols.append(second[:, -1])
    return torch.stack(cols, dim=1)
