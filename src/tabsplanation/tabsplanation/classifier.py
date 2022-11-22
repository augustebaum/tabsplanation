"""Declare, train and test the classifier model."""

from typing import List

import torch
from torch import nn

from tabsplanation.tabsplanation.autoencoder.architectures.base_ae import BaseModel


class Classifier(BaseModel):
    """A 3-layer MLP with ReLU activation throughout."""

    def __init__(
        self,
        output_dim: int,
        hidden_dims: List[int],
        learning_rate: float = 1e-3,
        model_name: str = None,
        **kwargs,
    ):
        super(Classifier, self).__init__(learning_rate)

        layers = nn.Sequential()
        for hidden_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.LazyLinear(out_features=hidden_dim),
                    nn.ReLU(),
                )
            )
        layers.append(nn.LazyLinear(out_features=output_dim))
        self.layers = layers

        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

    @classmethod
    def new(cls, **kwargs):
        return Classifier(**kwargs)

    def forward(self, X: torch.Tensor):
        return self.layers(X)

    def softmax(self, X: torch.Tensor):
        logits = self.layers(X)
        return torch.softmax(logits, dim=-1)

    def step(self, batch, batch_idx):
        x, y = batch
        y_predicted = self.layers(x)

        loss = self.loss_fn(y_predicted, y)
        logs = {"loss": loss}

        return loss, logs
