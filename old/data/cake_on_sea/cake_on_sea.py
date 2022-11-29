import json
import os
from datetime import datetime
from pathlib import Path

import click

import numpy as np


# Take out dead zone
def in_zone(points, zone):
    xs = points[:, 0]
    ys = points[:, 1]
    in_zone_x = (zone[0] <= xs) & (xs <= zone[1])
    in_zone_y = (zone[2] <= ys) & (ys <= zone[3])
    return in_zone_x & in_zone_y


def not_in_zone(points, zone):
    return np.logical_not(in_zone(points, zone))


def filter_zone(points, zone):
    """Take out the points in that are in the `zone`."""
    idx = np.where(not_in_zone(points, zone))[0]
    return points[idx]


def get_time():
    return datetime.now().strftime("%Y-%m-%dT%H:%M")


@click.command()
@click.option(
    "--seed",
    default=42,
    help="Seed for the randomness",
    show_default=True,
)
@click.option(
    "--gaussian/--uniform",
    default=False,
    help="Whether to generate points from a Gaussian or Uniform distribution",
    is_flag=True,
    show_default=True,
)
@click.option(
    "--nb_dims",
    default=250,
    help="Total number of columns in the dataset",
    show_default=True,
)
@click.option(
    "--nb_uncorrelated_dims",
    default=2,
    help="Number of uncorrelated columns (cannot be less than 2 or more than nb_dims)",
    show_default=True,
)
@click.option(
    "--nb_points_initial",
    default=100_000,
    help="Total number of rows in the dataset (will be slightly different in the end)",
    show_default=True,
)
@click.option(
    "--output_dir",
    default=get_time(),
    help="Output directory",
    show_default=True,
)
def main(seed, gaussian, nb_dims, nb_uncorrelated_dims, nb_points_initial, output_dir):

    np.random.seed(seed)
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
    points = filter_zone(points, dead_zone)
    # Recompute number of points
    nb_points = len(points)

    # Classification
    class_0 = [0, 50, 0, 25]
    class_1 = [0, 50, 25, 50]
    class_2 = [35, 45, 35, 45]

    # Class 0
    ys = np.zeros((len(points), 1), dtype=int)

    # Class 1
    ys[np.where(in_zone(points, class_1))[0]] = 1

    # Class 2
    ys[np.where(in_zone(points, class_2))[0]] = 2

    # Add correlated dims
    nb_correlated_dims = nb_dims - nb_uncorrelated_dims

    # TODO: Should the coefficients be distributed some way?
    coefs = rng.uniform(
        low=-10, high=10, size=(nb_uncorrelated_dims, nb_correlated_dims)
    )
    noise = rng.normal(loc=0, scale=1, size=(nb_points, nb_correlated_dims))
    # x_i = coef_i,1 * x_1 + coef_i,2 * x_2 + ... + coef_i,m * x_m + e_i
    correlated_features = points @ coefs + noise

    points = np.c_[points, correlated_features]

    dir = Path(output_dir)
    if not os.path.isdir(dir):
        os.mkdir(dir)

    metadata = {
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
    with open(dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    np.save(dir / "xs", points)
    np.save(dir / "ys", ys)
    np.save(dir / "coefs", coefs)


if __name__ == "__main__":
    main()
