import pickle
from typing import TypedDict

import torch
from omegaconf import OmegaConf

from config import BLD_PLOT_DATA
from experiments.shared.task_create_cake_on_sea import TaskCreateCakeOnSea
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import define_task, get_module_object, setup, Task
from tabsplanation.data import SyntheticDataset
from tabsplanation.models.autoencoder import AutoEncoder
from tabsplanation.models.classifier import Classifier
from tabsplanation.types import ExplanationPath, Tensor


def _get_method(method_name: str):
    return get_module_object("tabsplanation.explanations", method_name)


class PathResult(TypedDict):
    path: ExplanationPath
    valid: bool
    l1_distances_to_input: Tensor["nb_iterations", 1]
    likelihoods_nf: Tensor["nb_iterations", 1]


# This assumes the AutoEncoder is really a normalizing flow,
# and that `step` computes the log-likelihood.
def _make_path_result(
    classifier: Classifier, autoencoder: AutoEncoder, path: ExplanationPath
) -> PathResult:
    path_result = {}

    path.explained_input.x.detach_()
    path.explained_input.y.detach_()
    path.xs.detach_()
    path.ys.detach_()
    path_result["path"] = path

    # Get validities
    path_result["validity"] = int(path.target_class == classifier.predict(path.xs[-1]))

    def distance_metric(x_0, x_1):
        return (x_0 - x_1).norm(p=1, dim=-1)

    # Get distances
    # Note: Thanks, broadcasting!
    path_result["l1_distances_to_input"] = distance_metric(
        path.explained_input.x, path.xs
    ).detach()

    # Get likelihoods
    # TODO: ys are not necessary for autoencoder
    log_likelihoods = []
    for x in path.xs:
        nll, _ = autoencoder.step((x.reshape(1, -1), None), None)
        log_likelihoods.append(-nll)

    path_result["likelihoods_nf"] = torch.exp(torch.stack(log_likelihoods)).detach()

    # TODO: Measure time

    return path_result


class TaskCreatePlotDataCfPathMethods(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOT_DATA / "cf_path_methods"
        super(TaskCreatePlotDataCfPathMethods, self).__init__(cfg, output_dir)

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

        self.produces |= {"results": self.produces_dir / "results.pkl"}

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


define_task("compare_cf_methods", TaskCreatePlotDataCfPathMethods)
