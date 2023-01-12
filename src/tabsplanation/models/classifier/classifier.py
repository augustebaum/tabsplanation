"""Declare, train and test the classifier model."""

from typing import List, Optional

import torch
from torch import nn

from tabsplanation.models.base_model import BaseModel
from tabsplanation.types import RelativeFloat, StrictPositiveInt, Tensor


class Classifier(BaseModel):
    """An MLP with ReLU activation throughout, trained with Adam.

    The MLP has a variable number of hidden layers, and can optionally include
    Dropout before the ReLU and BatchNorm at the end of each layer.
    """

    def __init__(
        self,
        output_dim: StrictPositiveInt,
        hidden_dims: List[StrictPositiveInt],
        batch_norm: bool = False,
        dropout: Optional[RelativeFloat] = None,
        learning_rate: float = 1e-3,
        model_name: str = None,
        **kwargs,
    ):
        """Initializer.

        Inputs:
        -------
        * output_dim: The length of the tensor resulting from `forward`.
        * hidden_dims: The number of nodes in each hidden layer, in order.
        * batch_norm: Whether or not to apply a `BatchNorm` at the end of each hidden
            layer.
        * dropout: Whether or not to apply `Dropout` before the `ReLU` and if so, with
            what probability.
        * learning_rate: The learning rate of the optimizer.
        """
        super(Classifier, self).__init__(learning_rate)

        layers = nn.Sequential()
        for hidden_dim in hidden_dims:
            layer = [
                nn.LazyLinear(out_features=hidden_dim),
                nn.Dropout(dropout) if dropout else None,
                nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                nn.ReLU(),
            ]
            # Clean up the layer
            layer = [module for module in layer if module is not None]

            layers.append(nn.Sequential(*layer))

        layers.append(nn.LazyLinear(out_features=output_dim))

        self.layers = layers

        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, X: Tensor):
        return self.layers(X)

    def predict_proba(self, X: Tensor):
        logits = self.layers(X)
        return torch.softmax(logits, dim=-1)

    def predict(self, X: Tensor):
        return self.predict_proba(X).argmax(dim=-1)

    def step(self, batch, batch_idx):
        x, y = batch
        y_predicted = self.layers(x)

        loss = self.loss_fn(y_predicted, y)
        logs = {"loss": loss}

        return loss, logs
