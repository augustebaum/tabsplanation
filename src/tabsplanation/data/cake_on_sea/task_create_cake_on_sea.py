import json
from datetime import datetime

import numpy as np
import pytask

from tabsplanation.config import BLD

_produces_dir = BLD / "data" / "cake_on_sea" / "uniform"


@pytask.mark.produces(
    {
        "xs": _produces_dir / "xs.npy",
        "ys": _produces_dir / "ys.npy",
        "coefs": _produces_dir / "coefs.npy",
        "config": _produces_dir / "config.json",
    }
)
def task_create_cake_on_sea(produces):
    seed = 42
    gaussian = False
    nb_dims = 250
    nb_uncorrelated_dims = 2
    nb_points_initial = 100_000

    rng = np.random.default_rng(seed)

    # Uncorrelated dims
    if gaussian:
        points = rng.normal(
            loc=25, scale=25 / 3, size=(nb_points_initial, nb_uncorrelated_dims)
        )
    else:
        points = rng.uniform(
            low=0, high=50, size=(nb_points_initial, nb_uncorrelated_dims)
        )

    # Remove dead zone
    dead_zone = [35, 45, 25, 35]
    points = _filter_zone(points, dead_zone)
    # Recompute number of points
    nb_points = len(points)

    # Classification
    class_0 = [0, 50, 0, 25]
    class_1 = [0, 50, 25, 50]
    class_2 = [35, 45, 35, 45]

    # Class 0
    ys = np.zeros((len(points), 1), dtype=int)

    # Class 1
    ys[np.where(_in_zone(points, class_1))[0]] = 1

    # Class 2
    ys[np.where(_in_zone(points, class_2))[0]] = 2

    # Add correlated dims
    nb_correlated_dims = nb_dims - nb_uncorrelated_dims

    # TODO: Should the coefficients be distributed some way?
    coefficients = rng.uniform(
        low=-10, high=10, size=(nb_uncorrelated_dims, nb_correlated_dims)
    )
    noise = rng.normal(loc=0, scale=1, size=(nb_points, nb_correlated_dims))
    # x_i = coef_i,1 * x_1 + coef_i,2 * x_2 + ... + coef_i,m * x_m + e_i
    correlated_features = points @ coefficients + noise

    points = np.c_[points, correlated_features]

    metadata = {
        "generated_at": _get_time(),
        "seed": seed,
        "gaussian": gaussian,
        "class_0": class_0,
        "class_1": class_1,
        "class_2": class_2,
        "dead_zone": dead_zone,
        "nb_dims": nb_dims,
        "nb_uncorrelated_dims": nb_uncorrelated_dims,
        "nb_points_initial": nb_points_initial,
        "nb_points": nb_points,
    }
    with open(produces["config"], "w") as f:
        json.dump(metadata, f, indent=2)

    np.save(produces["coefs"], coefficients)
    np.save(produces["xs"], points)
    np.save(produces["ys"], ys)


def _get_time() -> str:
    return datetime.now().isoformat()


# Take out dead zone
def _in_zone(points, zone):
    xs = points[:, 0]
    ys = points[:, 1]
    in_zone_x = (zone[0] <= xs) & (xs <= zone[1])
    in_zone_y = (zone[2] <= ys) & (ys <= zone[3])
    return in_zone_x & in_zone_y


def _not_in_zone(points, zone):
    return np.logical_not(_in_zone(points, zone))


def _filter_zone(points, zone):
    """Take out the points in that are in the `zone`."""
    idx = np.where(_not_in_zone(points, zone))[0]
    return points[idx]
