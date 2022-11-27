"""Implementation of "NICE" from <https://paperswithcode.com/method/nice>.

Credits to <karim.hadidane@swisscom.com>.
"""
from typing import List, Optional

import torch
from torch import nn

from tabsplanation.autoencoder.architectures.base_ae import (
    BaseModel,
    ReconstructionLoss,
)
from tabsplanation.classifier import Classifier
from tabsplanation.normalizing_flow.coupling_layers import AdditiveCouplingLayer
from tabsplanation.normalizing_flow.losses import GaussianPriorLoss
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
        self.roundtrip_loss_fn = ReconstructionLoss()

        # Input dimension of MLP of a coupling layer is half the input dimension
        # TODO: What happens if the input dim is odd?
        half_dim = self.input_dim // 2

        # We use the same architecture in each layer
        def mlp():
            """Create a new untrained MLP."""
            return Classifier(
                output_dim=half_dim,
                hidden_dims=mlp_hidden_dims,
                batch_norm=batch_norm,
                dropout=dropout,
            )

        # We use 4 layers as it was done in the paper
        self.layers = nn.Sequential(
            AdditiveCouplingLayer("odd", mlp()),
            AdditiveCouplingLayer("even", mlp()),
            AdditiveCouplingLayer("odd", mlp()),
            AdditiveCouplingLayer("even", mlp()),
        )

        self.log_scaling_factors = nn.Parameter(torch.zeros(input_dim))

    def _run_step(self, x: Input) -> Latent:
        z = self.layers(x)
        z = z * torch.exp(self.log_scaling_factors)
        return z

    def forward(self, x: Input) -> Latent:
        return self._run_step(x)

    def inverse(self, z: Latent) -> Input:
        x = z * torch.exp(-1 * self.log_scaling_factors)
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def step(self, batch, batch_idx):
        x, y = batch
        z = self._run_step(x)

        roundtrip_loss = self.roundtrip_loss_fn(x, self.inverse(z))
        loss = self.loss_fn(z, self.log_scaling_factors)

        logs = {"loss": loss, "roundtrip_loss": roundtrip_loss}
        return loss, logs

    def encode(self, x: Input) -> Latent:
        return self._run_step(x)

    def decode(self, z: Latent) -> Input:
        return self.inverse(z)
