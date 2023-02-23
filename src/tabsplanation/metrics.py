"""Credits to

- <https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html> for the
MMD implementation, which I fixed to work without the assumption of equally sized
samples.
"""

from typing import Literal

import torch
from sklearn.metrics import auc as sklearn_auc
from sklearn.neighbors import LocalOutlierFactor

from tabsplanation.types import ExplanationPath, Tensor


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
