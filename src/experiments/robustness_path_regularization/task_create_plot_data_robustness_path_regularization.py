import random

import pandas as pd
import torch

from config import BLD_PLOT_DATA
from experiments.path_regularization.task_create_plot_data_path_regularization import (
    batchify_metrics,
    compute_mean_nll,
    get_explainer,
    get_loss_fn,
    TaskCreatePlotDataPathRegularization,
)
from experiments.path_regularization_cake_on_sea.task_train_path_regularized_ae import (
    TaskTrainPathRegAe,
)
from experiments.shared.data.task_get_data_module import TaskGetDataModule
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import clone_config, get_object_name, setup, Task
from tabsplanation.explanations.losses import get_path_mask

from tabsplanation.explanations.nice_path_regularized import random_targets_like
from tabsplanation.metrics import time_measurement
from tabsplanation.types import B, D, H, S, Tensor


def forward(
    self, autoencoder, cfs: Tensor[S, B, D], path_mask: Tensor[S, B], point: Tensor[D]
):
    s, b, d = cfs.shape
    distances_2d: Tensor[B * S] = torch.linalg.vector_norm(
        cfs.view(-1, d) - point, dim=-1
    )
    distances: Tensor[S, B] = distances_2d.view(s, b)
    import pdb

    pdb.set_trace()
    min_distances: Tensor[B] = distances.min(dim=-1).values
    return min_distances.mean()


class TaskCreatePlotDataRobustnessPathRegularization(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOT_DATA / "robustness_path_regularization"
        super(TaskCreatePlotDataRobustnessPathRegularization, self).__init__(
            cfg, output_dir
        )

        setup(cfg.seed)
        seeds = [random.randrange(100_000) for _ in range(cfg.nb_seeds)]

        # Make sure self.cfg is not modified
        cfg = clone_config(self.cfg)

        # For each dataset
        for data_module_cfg in cfg.data_modules:
            # Get the DataModule
            task_dataset = TaskGetDataModule.task_dataset(data_module_cfg)
            self.task_deps.append(task_dataset)

            # Train a classifier
            classifier_cfg = cfg.classifier
            classifier_cfg.data_module = data_module_cfg
            task_classifier = TaskTrainModel(classifier_cfg)

            # Add it to the dependencies
            self.task_deps.append(task_classifier)
            data_module_name = data_module_cfg.dataset.class_name
            self.depends_on[data_module_name] = {
                "dataset": task_dataset.produces,
                "classifier": task_classifier.produces,
                "autoencoders": {},
            }

            # Short-hand
            autoencoder_deps = self.depends_on[data_module_name]["autoencoders"]

            for seed in seeds:

                # Train an unregularized autoencoder
                autoencoder_cfg = cfg.autoencoder
                autoencoder_cfg.data_module = data_module_cfg
                autoencoder_cfg.seed = seed
                task_autoencoder = TaskTrainModel(autoencoder_cfg)

                # Add it to the dependencies
                self.task_deps.append(task_autoencoder)
                # Path regularized = False
                autoencoder_deps[(seed, False)] = task_autoencoder.produces

                for explainer_cfg in cfg.explainers:
                    path_regularized_autoencoder_cfg = clone_config(autoencoder_cfg)
                    path_regularized_autoencoder_cfg.model = {
                        "class_name": "PathRegularizedNICE",
                        "args": {
                            "classifier": classifier_cfg,
                            "autoencoder_args": autoencoder_cfg.model.args,
                            "explainer": explainer_cfg,
                            "hparams": cfg.path_reg.hparams,
                            "path_loss_fn": cfg.path_reg.path_loss_fn,
                        },
                    }

                    # Train a regularized autoencoder with the same architecture
                    task_path_regularized_autoencoder = TaskTrainPathRegAe(
                        path_regularized_autoencoder_cfg
                    )

                    # Add it to the dependencies
                    self.task_deps.append(task_path_regularized_autoencoder)
                    # Path regularized = True
                    autoencoder_deps[
                        (seed, True)
                    ] = task_path_regularized_autoencoder.produces

        self.produces |= {"results": self.produces_dir / "results.json"}

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        TaskCreatePlotDataPathRegularization._task_function(
            depends_on,
            produces,
            cfg,
            TaskCreatePlotDataRobustnessPathRegularization.get_path_results,
            TaskCreatePlotDataRobustnessPathRegularization.parse_result,
        )

    @staticmethod
    def parse_result(result):
        return {
            "Dataset": get_object_name(result["data_module"]).removesuffix("Dataset"),
            "Path method": result["path_method"]["name"],
            "Path regularization": result["path_regularized"],
            "Loss function": result["loss"]["name"],
            "validity_rate (%)": result["validity_rate"] * 100,
            r"\Delta t (ns)": result["time_per_path_step_s"] * (10 ** 9),
            "Mean NLL": result["mean_nll"],
        }

    @staticmethod
    def get_path_results(
        data_module, classifier, autoencoder, loss_fn, explainer, autoencoder_for_nll
    ):
        torch.cuda.empty_cache()

        loss_fn = get_loss_fn(loss_fn)

        explainer = get_explainer(classifier, autoencoder, explainer, loss_fn)

        batch_results = []

        for test_x, _ in data_module.test_dataloader(batch_size=200):
            metrics = {}

            test_x = test_x.to(classifier.device)

            y_predict = classifier.predict(test_x)
            target = random_targets_like(y_predict, data_module.dataset.output_dim)

            with time_measurement() as cf_time_s:
                cfs: Tensor[S, B, D] = explainer.get_cfs(test_x, target)

            nb_steps, nb_paths, _ = cfs.shape

            metrics["time_per_path_step_s"] = cf_time_s.time / (nb_paths * nb_steps)

            cf_preds = classifier.predict(cfs)

            path_mask: Tensor[S, B] = get_path_mask(cf_preds, target).to(
                cf_preds.device
            )

            metrics["validity_rate"] = (path_mask[0, :].sum() / nb_paths).item()

            metrics["mean_nll"] = compute_mean_nll(autoencoder_for_nll, cfs, path_mask)

            # metrics["mean_distance_to_max"] =

            batch_results.append(batchify_metrics(metrics, nb_paths))

        results = dict(
            pd.DataFrame.from_records(batch_results).sum() / len(data_module.test_set)
        )

        return results
