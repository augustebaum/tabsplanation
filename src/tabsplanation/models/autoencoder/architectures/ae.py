"""Implementation of a Variational Auto-Encoder using Lightning.

Credits to:
- <https://github.com/ethanluoyc/pytorch-vae>
- <https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py>
"""

from torch import nn

from .base_ae import AutoEncoder, LossFn, ReconstructionLoss
from .decoder import Decoder
from .encoder import Encoder


class AE(AutoEncoder):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        learning_rate: float = 1e-3,
        model_name: str = None,
        **kwargs,
    ):
        super(AE, self).__init__(encoder, decoder, learning_rate)

        self.loss_fn: LossFn = ReconstructionLoss()
        self.embedding = nn.Linear(encoder.hidden_dim, decoder.latent_dim)

    def encode(self, x):
        return self.embedding(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def _run_step(self, x):
        z = self.encode(x)
        return z, self.decoder(z)

    def forward(self, x):
        _, x_hat = self._run_step(x)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch
        _, x_hat = self._run_step(x)

        roundtrip_loss = self.loss_fn(x_hat, x)
        logs = {"roundtrip_loss": roundtrip_loss, "loss": roundtrip_loss}

        return roundtrip_loss, logs
