"""Credits to

- <https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html> for the
MMD implementation, which I fixed to work without the assumption of equally sized
samples."""

from typing import Literal

import torch
from sklearn.metrics import auc as sklearn_auc
from sklearn.neighbors import LocalOutlierFactor

from tabsplanation.types import ExplanationPath, Tensor


def mmd(x: Tensor["n", "D"], y: Tensor["m", "D"], kernel: Literal["multiscale", "rbf"]):
    """Empirical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Inputs:
    -------
    * x: Tensor containing sample of distribution P
    * y: Tensor containing sample of distribution Q
    * kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = x @ x.t(), y @ y.t(), x @ y.t()
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2.0 * yy  # Used for B in (1)
    dxy = xx.diag().reshape((-1, 1)) - 2.0 * zz + yy.diag()  # Used for C in (1)

    XX, YY, XY = (
        torch.zeros(xx.shape),
        torch.zeros(yy.shape),
        torch.zeros(zz.shape),
    )

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx) ** -1
            YY += a**2 * (a**2 + dyy) ** -1
            XY += a**2 * (a**2 + dxy) ** -1
    elif kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)
    else:
        raise NotImplementedError(f"Kernel {kernel} not recognized.")

    return XX.mean() + YY.mean() - 2.0 * XY.mean()


def train_lof(train_set):
    """
    See <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html>
    for more information.
    """
    lof = LocalOutlierFactor(novelty=True)
    lof.fit(train_set)
    return lof


def lof(trained_lof, path: ExplanationPath):
    """
    See <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html>
    for more information.
    """
    # Opposite of proper LOF, shifted by 1
    # From the docs:
    # The lower, the more abnormal.
    # Negative scores represent outliers, positive scores represent inliers.
    return trained_lof.decision_function(path.xs)


def auc(y):
    """Compute area-under-curve of a curve `y`.

    We assume that the curve starts at 0 and ends at 1.
    """
    x = torch.linspace(0, 1, steps=len(y))
    return sklearn_auc(x, y)
