"""Implementation of a Variational Auto-Encoder using Lightning.

Credits to:
- <https://github.com/ethanluoyc/pytorch-vae>
- <https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py>
"""
from typing import Optional, Tuple, TypeAlias

import torch
from torch import nn
from torchtyping import TensorType  # type: ignore

from tabsplanation.types import Input, Latent, Output, PositiveFloat
from .base_ae import AutoEncoder
from .decoder import Decoder, DecoderNormalized, DecoderSigmoid
from .encoder import Encoder

Mu: TypeAlias = TensorType["batch", "latent_dim"]
LogVar: TypeAlias = TensorType["batch", "latent_dim"]


class ELBOLoss(nn.Module):
    def __init__(
        self,
        beta: float,
        reduction: Optional[str] = "none",
    ) -> None:
        super(ELBOLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction="none")
        self.reduction = reduction
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
    ) -> torch.Tensor:
        reconstruction = self.loss_fn(y_pred, y_true).sum(-1)
        # <https://arxiv.org/abs/1312.6114>
        kl_divergence = -0.5 * (
            log_variance + 1 - mean.pow(2) - log_variance.exp()
        ).sum(-1)
        if self.reduction == "mean":
            reconstruction = reconstruction.mean()
            kl_divergence = kl_divergence.mean()
        return reconstruction + self.beta * kl_divergence, reconstruction, kl_divergence


class VAE(AutoEncoder):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        learning_rate: PositiveFloat = 1e-3,
        kl_factor: PositiveFloat = 1.0,
        model_name: str = None,
        **kwargs,
    ):
        super(VAE, self).__init__(encoder, decoder, learning_rate)

        # The encoder and decoder are generic NN layers.
        # The proper VAE machinery is right here.
        self.mu = nn.Linear(encoder.hidden_dim, decoder.latent_dim)
        self.log_var = nn.Linear(encoder.hidden_dim, decoder.latent_dim)

        self.kl_factor = kl_factor
        self.loss_fn = ELBOLoss(beta=kl_factor, reduction="mean")

        # This appends to the current list of hyperparameters
        self.save_hyperparameters("kl_factor")

    # <https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py>
    def forward(self, x: Input) -> Input:
        _, x_hat, _, _ = self._run_step(x)
        return x_hat

    def encode(self, x: Input) -> Latent:
        return self.mu(self.encoder(x))

    def decode(self, z: Latent) -> Input:
        return self.decoder(z)

    def _run_step(self, x: Input) -> Tuple[Latent, Input, Mu, LogVar]:
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        if self.training:
            z = self.sample(mu, log_var)
        else:
            z = mu
        return z, self.decoder(z), mu, log_var

    def sample(self, mu: Mu, log_var: LogVar) -> Latent:

        # Apply reparametrization trick to preserve gradient graph
        def reparametrize(mu: Mu, log_var: LogVar) -> Latent:
            stddev = torch.exp(log_var / 2)
            eps = torch.randn_like(stddev)
            return mu + eps * stddev

        z = reparametrize(mu, log_var)
        return z

    def step(self, batch: Tuple[Input, Output], batch_idx: int):
        x, y = batch
        z, x_hat, mu, log_var = self._run_step(x)

        loss, roundtrip_loss, kld = self.loss_fn(x_hat, x, mu, log_var)

        logs = {
            "roundtrip_loss": roundtrip_loss,
            "kld": kld,
            "loss": loss,
        }
        return loss, logs


class VAESigmoid(VAE):
    """The base VAE with a sigmoid and multiplication by 50 at the
    end of the decoder."""

    @classmethod
    def new(cls, **kwargs):
        return cls(
            encoder=Encoder(kwargs["input_dim"], kwargs["hidden_dim"]),
            decoder=DecoderSigmoid(
                kwargs["input_dim"], kwargs["hidden_dim"], kwargs["latent_dim"]
            ),
            **kwargs,
        )


class VAENormalized(VAE):
    """The base VAE with a Tanh and multiplication by 5 at the
    end of the decoder."""

    @classmethod
    def new(cls, **kwargs):
        return cls(
            encoder=Encoder(kwargs["input_dim"], kwargs["hidden_dim"]),
            decoder=DecoderNormalized(
                kwargs["input_dim"], kwargs["hidden_dim"], kwargs["latent_dim"]
            ),
            **kwargs,
        )
