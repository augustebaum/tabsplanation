# import pickle
import re
import sys

import pytask

from config import BLD_PLOT_DATA
from experiments.shared.task_create_cake_on_sea import TaskCreateCakeOnSea
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import get_configs, hash_, save_config, setup
from tabsplanation.data import SyntheticDataset


def camel_to_snake(str):
    """
    <https://www.geeksforgeeks.org/python-program-to-convert-camel-case-string-to-snake-case/>
    No shame!
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def get_module_object(module_path: str, object_name: str):
    """Import module given by `module_path` and return a function
    or class defined in that module with the name `object_name`."""
    exec(f"import {module_path}")
    return getattr(sys.modules[module_path], object_name)


def _get_method(method_name):
    return get_module_object("tabsplanation.explanations", method_name)


class TaskCreatePlotDataCfPathMethods:
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
        plot_data_dir = BLD_PLOT_DATA / "cf_path_methods" / self.id_
        self.produces = {
            "config": plot_data_dir / "config.yaml",
            "results": plot_data_dir / "results.csv",
        }

    @classmethod
    def task_function(cls, depends_on, produces, cfg):

        # cfg_plot = cfg.plot_data_class_2_paths

        device = setup(cfg.seed)

        dataset = SyntheticDataset(
            depends_on["xs"],
            depends_on["ys"],
            depends_on["coefs"],
            cfg.data.nb_dims,
            device,
        )
        # normalize = dataset.normalize
        # normalize_inverse = dataset.normalize_inverse

        results = {}

        for method_cfg in cfg.methods:
            method_class = _get_method(method_cfg.class_name)

            # for x in dataset:
            #     path = method.get_counterfactuals(x, y_target, clf, ae)
            #     measurements = _measure(path)
            #     results[method_cfg.class_name] = measurements

        # with open(produces["paths"], "wb") as paths_file:
        #     pickle.dump(paths, paths_file)

        save_config(cfg, produces["config"])


_task_class = TaskCreatePlotDataCfPathMethods
cfgs = get_configs("compare_cf_methods")

for cfg in cfgs:
    _task = _task_class(cfg)

    @pytask.mark.task(id=_task.id_)
    @pytask.mark.depends_on(_task.depends_on)
    @pytask.mark.produces(_task.produces)
    def task_create_plot_data_cf_path_methods(depends_on, produces, cfg=_task.cfg):
        _task_class.task_function(depends_on, produces, cfg)
