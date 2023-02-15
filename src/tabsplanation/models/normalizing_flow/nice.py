"""Implementation of "NICE" from <https://paperswithcode.com/method/nice>.

Credits to <karim.hadidane@swisscom.com>.
"""
from typing import List, Literal, Optional

import torch
from torch import nn

from tabsplanation.models.base_model import BaseModel
from tabsplanation.models.classifier import Classifier
from tabsplanation.models.normalizing_flow.coupling_layers import AdditiveCouplingLayer
from tabsplanation.models.normalizing_flow.losses import GaussianPriorLoss
from tabsplanation.types import (
    Input,
    Latent,
    PositiveFloat,
    RelativeFloat,
    StrictPositiveInt,
)


class NICEModel(BaseModel):
    """NICE model. Isn't it?"""

    def __init__(
        self,
        input_dim: StrictPositiveInt,
        learning_rate: PositiveFloat,
        mlp_hidden_dims: List[StrictPositiveInt],
        batch_norm: bool,
        dropout: Optional[RelativeFloat],
        **kwargs
    ):
        super(NICEModel, self).__init__(learning_rate)
        self.input_dim = input_dim

        self.loss_fn = GaussianPriorLoss()

        half_dim = self.input_dim // 2

        # We use the same architecture in each layer
        def mlp(partition: Literal["odd", "even"]):
            """Create a new untrained MLP."""
            output_dim = half_dim if partition == "even" else self.input_dim - half_dim
            return Classifier(
                output_dim=output_dim,
                hidden_dims=mlp_hidden_dims,
                batch_norm=batch_norm,
                dropout=dropout,
            )

        # We use 4 layers as it was done in the paper
        self.layers = nn.Sequential(
            AdditiveCouplingLayer("odd", mlp("odd")),
            AdditiveCouplingLayer("even", mlp("even")),
            AdditiveCouplingLayer("odd", mlp("odd")),
            AdditiveCouplingLayer("even", mlp("even")),
        )

        self.log_scaling_factors = nn.Parameter(
            torch.zeros(input_dim), requires_grad=True
        )

    def forward(self, x: Input) -> Latent:
        z = self.layers(x)
        z = z * torch.exp(self.log_scaling_factors)
        return z

    def inverse(self, z: Latent) -> Input:
        x = z * torch.exp(-1 * self.log_scaling_factors)
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def step(self, batch, batch_idx):
        x, y = batch
        z = self(x)

        loss = self.loss_fn(z, self.log_scaling_factors)

        logs = {
            "loss": loss,
        }
        return loss, logs

    def encode(self, x: Input) -> Latent:
        return self(x)

    def decode(self, z: Latent) -> Input:
        return self.inverse(z)
