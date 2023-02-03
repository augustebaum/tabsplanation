import pickle
import time
from typing import TypedDict

import torch
from omegaconf import OmegaConf
from sklearn.neighbors import LocalOutlierFactor

from config import BLD_PLOT_DATA
from experiments.shared.data.task_create_cake_on_sea import TaskCreateCakeOnSea
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import (
    define_task,
    get_data_module,
    get_module_object,
    setup,
    Task,
)
from tabsplanation.data import CakeOnSeaDataset
from tabsplanation.metrics import auc, lof, train_lof
from tabsplanation.models.autoencoder import AutoEncoder
from tabsplanation.models.classifier import Classifier
from tabsplanation.types import ExplanationPath, Tensor


def _get_method(method_name: str):
    return get_module_object("tabsplanation.explanations", method_name)


def instantiate_method(method_cfg, depends_on):
    # Recover the class from its name
    method_class = _get_method(method_cfg.class_name)

    # Instantiate method
    autoencoder = torch.load(
        depends_on[f"autoencoder_{method_cfg.class_name}"]["model"]
    )
    classifier = torch.load(depends_on[f"classifier_{method_cfg.class_name}"]["model"])
    kwargs = OmegaConf.to_object(method_cfg.args) | {
        "autoencoder": autoencoder,
        "classifier": classifier,
    }
    method = method_class(**kwargs)

    return method, classifier, autoencoder


class PathResult(TypedDict):
    path: ExplanationPath
    validity: bool
    l1_distances_to_input: Tensor["nb_iterations", 1]
    likelihoods_nf: Tensor["nb_iterations", 1]
    runtime_per_step_milliseconds: float

    # This assumes the AutoEncoder is really a normalizing flow,
    # and that `step` computes the log-likelihood.
    @classmethod
    def new(
        cls,
        dataset: CakeOnSeaDataset,
        trained_lof: LocalOutlierFactor,
        classifier: Classifier,
        autoencoder: AutoEncoder,
        path: ExplanationPath,
        duration_per_step_ns: int,
    ) -> "PathResult":
        path_result = {}

        path.explained_input.input = path.explained_input.input.detach()
        path.explained_input.output = path.explained_input.output.detach()
        path.explained_input.input = dataset.normalize_inverse(path.explained_input.x)
        path.xs.detach_()
        path.xs = dataset.normalize_inverse(path.xs)
        path.ys.detach_()

        path_result["path"] = path

        # Get validities
        path_result["validity"] = int(
            path.target_class == classifier.predict(path.xs[-1])
        )

        def distance_metric(x_0, x_1):
            return (x_0 - x_1).norm(p=1, dim=-1)

        # Get distances
        # Note: Thanks, broadcasting!
        path_result["l1_distances_to_input"] = distance_metric(
            path.explained_input.x, path.xs
        ).detach()
        path_result["l1_distances_to_input_auc"] = auc(
            path_result["l1_distances_to_input"]
        )

        # Get likelihoods
        log_likelihoods = []
        for x in path.xs:
            # Don't include the scaling factors
            nll = autoencoder.layers(x.reshape(1, -1)).squeeze()
            log_likelihoods.append(-nll)

        path_result["likelihoods_nf"] = torch.exp(torch.stack(log_likelihoods)).detach()
        # path_result["likelihoods_nf_auc"] = auc(path_result["likelihoods_nf"])

        # Too complicated
        # Get MMD
        # mmds_global = torch.stack(
        #     [mmd(x_.reshape((1, -1)), dataset.X, "rbf") for x_ in path.xs]
        # )
        # path_result["mmds_global"] = mmds_global

        # # Instead of computing the MMD with the whole dataset, compute it for the subset of
        # # points that have the same class as the counterfactual
        # dataset_X_class = {c: dataset.X[dataset.y == c] for c in dataset.y.unique()}

        # mmds_class = torch.stack(
        #     [
        #         mmd(x_.reshape((1, -1)), dataset_X_class[y_], "rbf")
        #         for x_, y_ in zip(path.xs, path.ys)
        #     ]
        # )
        # path_result["mmds_global"] = mmds_class

        path_result["lof"] = lof(trained_lof, path)
        path_result["lof_auc"] = auc(path_result["lof"])

        path_result["runtime_per_step_milliseconds"] = duration_per_step_ns / 1_000_000

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

        data_module = get_data_module(depends_on, cfg, device)

        full_dataset = data_module.dataset
        trained_lof = train_lof(full_dataset.X)

        # test_loader = data_module.test_dataloader()
        # xs, ys = next(iter(test_loader))
        # xs, ys = xs[:5], ys[:5]

        # Get some test points
        test_point_indices = (
            data_module.test_set.indices[: cfg.nb_test_points]
            if cfg.nb_test_points is not None
            else data_module.test_set.indices
        )
        xs, ys = full_dataset[test_point_indices]

        results = {}

        for method_cfg in cfg.methods:

            results[method_cfg.class_name] = []
            method, classifier, autoencoder = instantiate_method(method_cfg, depends_on)

            for x in xs:

                y_pred = classifier.predict(x)
                # 0 goes to 1, 1 goes to 2, 2 goes to 0
                y_target = (y_pred + 1) % 3

                # Measure running time
                start_time_ns = time.time_ns()
                path = method.get_counterfactuals(x, y_target)
                call_duration_ns = time.time_ns() - start_time_ns
                duration_per_step_ns = call_duration_ns / len(path)

                path_result = PathResult.new(
                    full_dataset,
                    trained_lof,
                    classifier,
                    autoencoder,
                    path,
                    duration_per_step_ns,
                )
                results[method_cfg.class_name].append(path_result)

        with open(produces["results"], "wb") as paths_file:
            pickle.dump(results, paths_file)


# task, task_definition = define_task(
#     "compare_cf_methods", TaskCreatePlotDataCfPathMethods
# )
# exec(task_definition)
