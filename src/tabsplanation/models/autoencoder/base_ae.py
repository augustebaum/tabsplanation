from torch import nn

from tabsplanation.models.base_model import BaseModel

from tabsplanation.types import Tensor
from .decoder import Decoder
from .encoder import Encoder


class ReconstructionLoss(nn.Module):
    """A adapted MSELoss that takes the average over each row error,
    rather than over all elements."""

    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.loss = nn.MSELoss(reduction="sum")

    def forward(
        self,
        x: Tensor["batch", "input_dim"],
        x_hat: Tensor["batch", "input_dim"],
    ):
        return self.loss(x, x_hat) / len(x)


class AutoEncoder(BaseModel):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        learning_rate: float,
    ):
        assert encoder.input_dim == decoder.input_dim, (
            "Input dimensions must be the same for the encoder and for the decoder."
            f"The input dimensions for the encoder are {encoder.input_dim} "
            f"and for the decoder they are {decoder.input_dim}."
        )

        super(AutoEncoder, self).__init__(learning_rate)

        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def new(cls, **kwargs):
        return cls(
            encoder=Encoder(kwargs["input_dim"], kwargs["hidden_dim"]),
            decoder=Decoder(
                kwargs["input_dim"], kwargs["hidden_dim"], kwargs["latent_dim"]
            ),
            **kwargs,
        )
