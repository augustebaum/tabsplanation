import hashlib
import json
from typing import Type, TypeAlias

import pytorch_lightning as pl
from torch import nn, optim
from torchtyping import TensorType

from .decoder import Decoder
from .encoder import Encoder


# PyTorch loss function
LossFn: TypeAlias = Type["_LossFn"]


class BaseModel(pl.LightningModule):
    def __init__(self, learning_rate: float, model_name: str = None):
        super(BaseModel, self).__init__()

        self.learning_rate = learning_rate
        self.model_name = model_name or self.__class__.__name__

        self.save_hyperparameters("learning_rate")

    @property
    def model_dict(self) -> dict:
        modules = {k: repr(v) for k, v in self._modules.items()}
        return {
            "model_name": self.model_name,
            "modules": modules,
            "hparams": self.hparams,
        }

    def get_code(self) -> str:
        """Combine all the information about the model and
        hyperparameters into a unique identifier."""
        # Produce a string from the model_dict reproducibly
        model_str = json.dumps(self.model_dict, sort_keys=True, ensure_ascii=True)
        # Hash the resulting string
        hash = hashlib.sha256(model_str.encode("ascii")).hexdigest()
        # Maybe ambitious?
        return hash[:6]

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})
        return loss


class ReconstructionLoss(nn.Module):
    """A adapted MSELoss that takes the average over each row error,
    rather than over all elements."""

    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.loss = nn.MSELoss(reduction="sum")

    def forward(
        self,
        x: TensorType["batch", "input_dim"],
        x_hat: TensorType["batch", "input_dim"],
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
