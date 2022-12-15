import pickle
from typing import Any

import pytask
import torch
from omegaconf import OmegaConf

from config import BLD_PLOT_DATA
from experiments.shared.task_create_cake_on_sea import TaskCreateCakeOnSea
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import (
    get_configs,
    get_module_object,
    hash_,
    save_config,
    save_full_config,
    setup,
)
from tabsplanation.data import SyntheticDataset
from tabsplanation.models.autoencoder import AutoEncoder
from tabsplanation.models.classifier import Classifier
from tabsplanation.types import ExplanationPath


def _get_method(method_name: str):
    return get_module_object("tabsplanation.explanations", method_name)


PathResult = Any

# This assumes the AutoEncoder is really a normalizing flow
# with Gaussian prior.
def realness_score(autoencoder: AutoEncoder, x):
    z = autoencoder.encode(x)
    # log_density =


# This assumes the AutoEncoder is really a normalizing flow,
# and that `step` computes the log-likelihood.
def _make_path_result(
    classifier: Classifier, autoencoder: AutoEncoder, path: ExplanationPath
) -> PathResult:
    path_result = {}

    path.explained_input.x.detach_()
    path.explained_input.y.detach_()
    path_result["explained_input"] = path.explained_input
    path_result["target_class"] = path.target_class
    path_result["cf_xs"] = path.xs.detach()
    path_result["cf_ys"] = path.ys.detach()

    # Get validities
    path_result["validity"] = int(path.target_class == classifier.predict(path.xs[-1]))

    def distance_metric(x_0, x_1):
        return (x_0 - x_1).norm(p=1, dim=-1)

    # Get distances
    # Note: Thanks, broadcasting!
    path_result["distances_to_input"] = distance_metric(
        path.explained_input.x, path.xs
    ).detach()

    # Get likelihoods
    # TODO: ys are not necessary
    log_likelihoods = []
    for x in path.xs:
        nll, _ = autoencoder.step((x.reshape(1, -1), None), None)
        log_likelihoods.append(-nll)

    path_result["likelihoods_nf"] = torch.exp(torch.stack(log_likelihoods)).detach()

    # TODO: Measure time

    return path_result


class TaskCreatePlotDataCfPathMethods:
    def __init__(self, cfg):
        self.cfg = cfg

        task_create_cake_on_sea = TaskCreateCakeOnSea(self.cfg)

        self.depends_on = task_create_cake_on_sea.produces

        for method in self.cfg.methods:
            task_train_autoencoder = TaskTrainModel(method.args.autoencoder)
            task_train_classifier = TaskTrainModel(method.args.classifier)

            self.depends_on = (
                self.depends_on
                | {f"autoencoder_{method.class_name}": task_train_autoencoder.produces}
                | {f"classifier_{method.class_name}": task_train_classifier.produces}
            )

        self.id_ = hash_(self.cfg)
        plot_data_dir = BLD_PLOT_DATA / "cf_path_methods" / self.id_
        self.produces = {
            "config": plot_data_dir / "config.yaml",
            "full_config": plot_data_dir / "full_config.yaml",
            "results": plot_data_dir / "results.pkl",
        }

    @classmethod
    def task_function(cls, depends_on, produces, cfg):

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
            # Recover the class from its name
            method_class = _get_method(method_cfg.class_name)
            results[method_cfg.class_name] = []

            # Instantiate method
            autoencoder = torch.load(
                depends_on[f"autoencoder_{method_cfg.class_name}"]["model"]
            )
            classifier = torch.load(
                depends_on[f"classifier_{method_cfg.class_name}"]["model"]
            )
            kwargs = OmegaConf.to_object(method_cfg.args) | {
                "autoencoder": autoencoder,
                "classifier": classifier,
            }
            method = method_class(**kwargs)

            # TODO: Use test loader
            xs, ys = dataset[:5]
            # for i, x in test_data:
            for x in xs:
                # path = method.get_counterfactuals(x, target_map[i])
                y_pred = classifier.predict(x)
                y_target = (y_pred + 1) % 3
                # TODO: Use context to measure time taken
                path = method.get_counterfactuals(x, y_target)

                path_result = _make_path_result(classifier, autoencoder, path)
                results[method_cfg.class_name].append(path_result)

        with open(produces["results"], "wb") as paths_file:
            pickle.dump(results, paths_file)


cfgs = get_configs("compare_cf_methods")
_task_class = TaskCreatePlotDataCfPathMethods

for cfg in cfgs:
    task = _task_class(cfg)

    @pytask.mark.task(id=task.id_)
    @pytask.mark.depends_on(task.depends_on)
    @pytask.mark.produces(task.produces)
    def task_create_plot_data_cf_path_methods(depends_on, produces, cfg=task.cfg):
        _task_class.task_function(depends_on, produces, cfg)
        save_full_config(cfg, produces["full_config"])
        save_config(cfg, produces["config"])
