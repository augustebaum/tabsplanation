import pickle
import sys

import numpy as np
import pytask
import torch

from config import BLD_PLOT_DATA
from experiments.shared.data.task_create_cake_on_sea import TaskCreateCakeOnSea
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import get_configs, hash_, save_config, setup
from tabsplanation.data import SyntheticDataset


class TaskCreatePlotDataClass2Paths:
    def __init__(self, cfg):
        self.cfg = cfg

        task_create_cake_on_sea = TaskCreateCakeOnSea(self.cfg)
        task_train_autoencoder = TaskTrainModel(self.cfg.models.autoencoder)
        task_train_classifier = TaskTrainModel(self.cfg.models.classifier)

        self.depends_on = (
            task_create_cake_on_sea.produces
            | {"autoencoder": task_train_autoencoder.produces}
            | {"classifier": task_train_classifier.produces}
        )

        self.id_ = hash_(self.cfg)
        plot_data_dir = BLD_PLOT_DATA / "class_2_paths" / self.id_
        self.produces = {
            "config": plot_data_dir / "config.yaml",
            "paths": plot_data_dir / "paths.pkl",
        }


cfgs = get_configs("latent_shift")

for cfg in cfgs:
    task = TaskCreatePlotDataClass2Paths(cfg)

    @pytask.mark.task(id=task.id_)
    @pytask.mark.depends_on(task.depends_on)
    @pytask.mark.produces(task.produces)
    def task_create_plot_data_class_2_paths(depends_on, produces, cfg=task.cfg):

        cfg_plot = cfg.plot_data_class_2_paths

        device = setup(cfg.seed)

        dataset = SyntheticDataset(
            depends_on["xs"],
            depends_on["ys"],
            depends_on["coefs"],
            cfg.data.nb_dims,
            device,
        )
        normalize = dataset.normalize
        normalize_inverse = dataset.normalize_inverse

        inputs_denorm = []

        # Cover class 2 (4 corners and middle)
        margin = 2
        nb_points = cfg_plot.nb_points
        inputs_denorm = torch.tensor(
            np.c_[
                np.linspace(35 + margin, 45 - margin, num=nb_points),
                np.ones(nb_points) * 43,
            ],
            dtype=torch.float,
        )
        inputs_denorm = dataset.fill_from_2d_point(inputs_denorm)

        # torch.set_printoptions(precision=3, sci_mode=False)
        inputs = normalize(inputs_denorm)

        clf = torch.load(depends_on["classifier"]["model"])
        ae = torch.load(depends_on["autoencoder"]["model"])
        make_path = _get_make_path_fn(cfg_plot.path_algorithm)
        paths = [
            make_path(input=input, target_class=0, clf=clf, ae=ae) for input in inputs
        ]

        for path in paths:
            path.explained_input.input = normalize_inverse(path.explained_input.input)
            path.xs = normalize_inverse(path.xs)

        with open(produces["paths"], "wb") as paths_file:
            pickle.dump(paths, paths_file)

        save_config(cfg, produces["config"])


def _get_make_path_fn(function_name: str):
    import tabsplanation.explanations.latent_shift  # ignore

    return getattr(
        sys.modules["tabsplanation.explanations.latent_shift"], function_name
    )
