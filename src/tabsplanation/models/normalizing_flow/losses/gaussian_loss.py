import numpy as np
import torch
import torch.nn as nn

from tabsplanation.types import B, D, Latent, Tensor

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
        if avg:
            self.agg = torch.mean
        else:
            self.agg = torch.sum

    @classmethod
    def log_likelihood(cls, z: Tensor[B, D]) -> Tensor[B]:
        """Compute the log-likelihood for a point or batch of latent points,
        with a standard Gaussian prior.
        """
        constant = -z.shape[1] * 0.5 * _log_two_pi
        return -0.5 * (z ** 2).sum(dim=1) + constant

    def forward(self, z: Latent, log_scaling_factors: Tensor[D]):
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
        return self.agg(nll)
