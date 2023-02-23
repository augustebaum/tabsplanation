"""Credits to

- <https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html> for the
MMD implementation, which I fixed to work without the assumption of equally sized
samples.
"""

from functools import lru_cache
from time import perf_counter

import torch
from sklearn.metrics import auc as sklearn_auc
from sklearn.neighbors import LocalOutlierFactor

from tabsplanation.types import D, Tensor


@lru_cache()
def train_lof(train_set: Tensor):
    """
    See <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html>
    for more information.
    """
    data = train_set.cpu().numpy()
    lof = LocalOutlierFactor(novelty=True)
    lof.fit(data)
    return lof


def lof(trained_lof, points: Tensor[..., D]):
    """
    Compute the opposite of proper LOF, shifted by 1. From the docs:
    - Negative scores represent outliers, positive scores represent inliers.
    - The lower, the more abnormal.

    See <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html>
    for more information.

    Inputs:
    -------
    * points: The points for which the LOF is to be computed.
    * trained_lof: A `LocalOutlierFactor` instance trained on a set of points
        of the same dimensionality as `points`.
    """
    points_np = points.view(-1, points.shape[-1]).detach().cpu().numpy()
    lof_np = trained_lof.decision_function(points_np)
    return torch.from_numpy(lof_np).to(points.device).view(points.shape[:-1])


def auc(y):
    """Compute area-under-curve of a curve `y`.

    We assume that the curve starts at 0 and ends at 1.
    """
    x = torch.linspace(0, 1, steps=len(y))
    return sklearn_auc(x, y)


# <https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time>
class time_measurement:
    """Context to measure the time it takes to run a code block, in seconds."""

    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        self.readout = f"Time: {self.time:.3f} seconds"
