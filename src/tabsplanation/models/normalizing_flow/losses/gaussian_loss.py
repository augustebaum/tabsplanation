import numpy as np
import torch
import torch.nn as nn

from tabsplanation.types import Latent, Tensor

_log_two_pi = torch.log(torch.tensor(2 * np.pi))


class GaussianPriorLoss(nn.Module):
    """Implements the log-likelihood criterion presented in the NICE paper,
    with a standard Gaussian distribution as the latent distribution prior.
    """

    def __init__(self, avg=True):
        """Initialize.

        Inputs:
        -------
        * avg: Whether or not to average the negative log likelihood (as
            opposed to just summing).
        """
        super(GaussianPriorLoss, self).__init__()
        self.avg = avg

    @classmethod
    def log_likelihood(cls, z: Tensor["batch", "input_dim"]) -> Tensor["batch"]:
        """Compute the log-likelihood for a point or batch of latent points,
        with a standard Gaussian prior.
        """
        d = z.shape[1]
        constant = -d * 0.5 * _log_two_pi
        return -0.5 * (z**2).sum(dim=1) + constant

    def forward(self, z: Latent, log_scaling_factors: Tensor["input_dim"]):
        """Compute the sum or average negative log-likelihood.

        The goal is to maximize the likelihood, which is equivalent to minimizing
        the log-likelihood.
        The scaling factors are expected to be logged already, as in the
        architecture used in the original NICE paper.

        Inputs:
        -------
        * z: The point or batch of points in latent space whose likelihood
            will be computed.
        * log_scaling_factors: The logarithm of the learned scaling factors.

        Returns:
        --------
        The mean or total negative log-likelihood over the batch, depending on how the
        loss was initialized.
        """
        nll = -(GaussianPriorLoss.log_likelihood(z) + log_scaling_factors.sum())
        if self.avg:
            return torch.mean(nll)
        return torch.sum(nll)
