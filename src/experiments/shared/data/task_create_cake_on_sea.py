import json
from typing import TypedDict

import numpy as np

# import pytask
# from omegaconf import OmegaConf

from config import BLD_DATA
from experiments.shared.utils import get_configs, get_time, hash_, save_config, Task
from tabsplanation.types import PositiveInt


class CreateCakeOnSeaCfg(TypedDict):
    seed: int
    gaussian: bool
    nb_points_initial: PositiveInt
    nb_uncorrelated_dims: PositiveInt
    nb_dims: PositiveInt


class TaskCreateCakeOnSea(Task):
    def __init__(self, cfg):
        output_dir = BLD_DATA / "cake_on_sea"
        super(TaskCreateCakeOnSea, self).__init__(cfg, output_dir)

        self.produces |= {
            "xs": self.produces_dir / "xs.npy",
            "ys": self.produces_dir / "ys.npy",
            "coefs": self.produces_dir / "coefs.npy",
            "metadata": self.produces_dir / "metadata.json",
        }

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        import pdb

        pdb.set_trace()
        rng = np.random.default_rng(cfg.seed)

        # Uncorrelated dims
        if cfg.gaussian:
            points = rng.normal(
                loc=25,
                scale=25 / 3,
                size=(cfg.nb_points_initial, cfg.nb_uncorrelated_dims),
            )
        else:
            points = rng.uniform(
                low=0, high=50, size=(cfg.nb_points_initial, cfg.nb_uncorrelated_dims)
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
        nb_correlated_dims = cfg.nb_dims - cfg.nb_uncorrelated_dims

        # TODO: Should the coefficients be distributed some way?
        coefficients = rng.uniform(
            low=-10, high=10, size=(cfg.nb_uncorrelated_dims, nb_correlated_dims)
        )
        noise = rng.normal(loc=0, scale=1, size=(nb_points, nb_correlated_dims))
        # x_i = coef_i,1 * x_1 + coef_i,2 * x_2 + ... + coef_i,m * x_m + e_i
        correlated_features = points @ coefficients + noise

        points = np.c_[points, correlated_features]

        metadata = {
            "generated_at": get_time(),
            "class_0": class_0,
            "class_1": class_1,
            "class_2": class_2,
            "dead_zone": dead_zone,
            "nb_points": nb_points,
            **cfg,
        }
        with open(produces["metadata"], "w") as f:
            json.dump(metadata, f, indent=2)

        np.save(produces["coefs"], coefficients)
        np.save(produces["xs"], points)
        np.save(produces["ys"], ys)

        # save_config(cfg, produces["config"])


# cfgs = get_configs()
# cfgs = []

# for cfg in cfgs:
#     task = TaskCreateCakeOnSea(cfg)

#     @pytask.mark.task(id=task.id_)
#     @pytask.mark.produces(task.produces)
#     def task_create_cake_on_sea(produces, cfg=task.cfg):
#         TaskCreateCakeOnSea.task_function(None, produces, cfg)


# Take out dead zone
def _in_zone(points, zone):
    xs = points[:, 0]
    ys = points[:, 1]
    in_zone_x = (zone[0] <= xs) & (xs <= zone[1])
    in_zone_y = (zone[2] <= ys) & (ys <= zone[3])
    return in_zone_x & in_zone_y


def _not_in_zone(points, zone):
    return ~_in_zone(points, zone)


def _filter_zone(points, zone):
    """Take out the points in `points` that are in the `zone`."""
    return points[_not_in_zone(points, zone)]
