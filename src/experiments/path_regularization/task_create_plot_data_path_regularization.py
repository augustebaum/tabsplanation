import itertools
import random

import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from config import BLD_PLOT_DATA
from experiments.path_regularization_cake_on_sea.task_train_path_regularized_ae import (
    TaskTrainPathRegAe,
)
from experiments.shared.data.task_get_data_module import TaskGetDataModule
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import (
    clone_config,
    get_object,
    parse_full_qualified_object,
    read,
    setup,
    Task,
    write,
)
from tabsplanation.explanations.losses import (
    get_path_ends,
    get_path_mask,
    where_changes,
)
from tabsplanation.explanations.nice_path_regularized import random_targets_like
from tabsplanation.metrics import time_measurement
from tabsplanation.types import B, D, H, S, Tensor, V


def batchify_metrics(metrics, batch_size):
    new_metrics = metrics.copy()
    for k, v in metrics.items():
        metrics[k] = v * batch_size
    return new_metrics


def get_explainer(classifier, autoencoder, explainer_cfg, loss_fn):
    explainer_cls = get_object(explainer_cfg.class_name)
    explainer_hparams = explainer_cfg.args.hparams

    return explainer_cls(classifier, autoencoder, explainer_hparams, loss_fn)


def get_loss_fn(loss_fn_cfg):
    loss_cls = get_object(loss_fn_cfg.class_name)
    if loss_fn_cfg.args is None:
        return loss_cls()
    return loss_cls(**loss_fn_cfg.args)


def compute_mean_nll(autoencoder, valid_paths: Tensor[S, V, D], path_mask):
    nb_steps, nb_valid = valid_paths.shape[0], valid_paths.shape[1]
    if nb_valid == 0:
        mean_nll = 0
    else:
        z: Tensor[S * V, H] = autoencoder(valid_paths.view(-1, valid_paths.shape[-1]))
        nll: Tensor[S * V] = -(
            autoencoder.loss_fn.__class__.log_likelihood(z)
            + autoencoder.log_scaling_factors.sum()
        )

        masked_nll = path_mask * nll.view(nb_steps, nb_valid, 1)
        mean_nll = masked_nll.mean().item()

    return mean_nll


def compute_mean_boundary_crossings_rate(
    valid_cf_preds: Tensor[V, S], path_mask: Tensor[V, S]
):
    nb_valid = path_mask.shape[0]
    if nb_valid == 0:
        return 0
    bound_crossings_before_target: Tensor[V, S] = (
        where_changes(valid_cf_preds) * path_mask
    )
    return bound_crossings_before_target.sum(dim=1).mean().item()


class TaskCreatePlotDataPathRegularization(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOT_DATA / "path_regularization"
        super(TaskCreatePlotDataPathRegularization, self).__init__(cfg, output_dir)
        self.task_deps, self.depends_on = TaskCreatePlotDataPathRegularization.setup(
            cfg
        )
        self.produces |= {"results": self.produces_dir / "results.json"}

    @staticmethod
    def setup(cfg):
        cfg = clone_config(cfg)

        setup(cfg.seed)
        seeds = [random.randrange(100_000) for _ in range(cfg.nb_seeds)]

        task_deps = []
        depends_on = {}

        # Make sure self.cfg is not modified

        # For each dataset
        for data_module_cfg in cfg.data_modules:
            # Get the DataModule
            task_dataset = TaskGetDataModule.task_dataset(data_module_cfg)
            task_deps.append(task_dataset)

            # Train a classifier
            classifier_cfg = cfg.classifier
            classifier_cfg.data_module = data_module_cfg
            task_classifier = TaskTrainModel(classifier_cfg)

            # Train an unregularized autoencoder on the master seed for computing NLL
            autoencoder_cfg = cfg.autoencoder
            autoencoder_cfg.data_module = data_module_cfg
            autoencoder_cfg.seed = cfg.seed
            task_autoencoder = TaskTrainModel(autoencoder_cfg)

            # Add it to the dependencies
            task_deps.append(task_classifier)
            task_deps.append(task_autoencoder)
            data_module_name = data_module_cfg.dataset.class_name
            depends_on[data_module_name] = {
                "dataset": task_dataset.produces,
                "classifier": task_classifier.produces,
                "autoencoder": task_autoencoder.produces,
                "autoencoders": {},
            }

            # Short-hand
            autoencoder_deps = depends_on[data_module_name]["autoencoders"]

            for seed in seeds:

                # Train an unregularized autoencoder per seed
                autoencoder_cfg = cfg.autoencoder
                autoencoder_cfg.data_module = data_module_cfg
                autoencoder_cfg.seed = seed
                task_autoencoder = TaskTrainModel(autoencoder_cfg)

                # Add it to the dependencies
                task_deps.append(task_autoencoder)
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
                    task_deps.append(task_path_regularized_autoencoder)
                    # Path regularized = True
                    autoencoder_deps[
                        (seed, True)
                    ] = task_path_regularized_autoencoder.produces

        return task_deps, depends_on

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        TaskCreatePlotDataPathRegularization._task_function(
            depends_on,
            produces,
            cfg,
            TaskCreatePlotDataPathRegularization.get_path_results,
            TaskCreatePlotDataPathRegularization.parse_result,
        )

    @classmethod
    def _task_function(
        cls, depends_on, produces, cfg, get_path_results_fn, parse_result_fn
    ):
        results = []
        for data_module_name, values in depends_on.items():
            device = setup(cfg.seed)

            classifier_cfg = OmegaConf.load(values["classifier"]["full_config"])

            data_module = TaskGetDataModule.read_data_module(
                values["dataset"], classifier_cfg.data_module, device
            )

            print(f"Dataset: {data_module.dataset.__class__.__name__}")

            classifier = read(values["classifier"]["model"], device=device)
            autoencoder_for_nll = read(values["autoencoder"]["model"], device=device)

            for (seed, path_regularized), autoencoder_path in values[
                "autoencoders"
            ].items():
                print(f"Seed: {seed}")
                autoencoder = read(autoencoder_path["model"], device=device)

                for path_method, loss_fn in tqdm(
                    list(itertools.product(cfg.explainers, cfg.losses))
                ):
                    path_results = get_path_results_fn(
                        data_module,
                        classifier,
                        autoencoder,
                        loss_fn,
                        path_method,
                        autoencoder_for_nll,
                    )

                    result = parse_result_fn(
                        {
                            "data_module": data_module_name,
                            "path_method": path_method,
                            "seed": seed,
                            "path_regularized": path_regularized,
                            "loss": loss_fn,
                            **path_results,
                        }
                    )

                    results.append(result)

        write(results, produces["results"])

    @staticmethod
    def parse_result(result):
        get_object_name = lambda s: parse_full_qualified_object(s)[1]
        return {
            "Dataset": get_object_name(result["data_module"]).removesuffix("Dataset"),
            "Path method": result["path_method"]["name"],
            "Path regularization": result["path_regularized"],
            "Loss function": result["loss"]["name"],
            "validity_rate (%)": result["validity_rate"] * 100,
            r"\Delta t (ns)": result["time_per_iteration_s"] * (10 ** 9),
            "Mean #BC": result["mean_boundary_crossings"],
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

        for test_x, _ in data_module.test_dataloader():
            metrics = {}

            test_x = test_x.to(classifier.device)

            y_predict = classifier.predict(test_x)
            target = random_targets_like(y_predict, data_module.dataset.output_dim)

            with time_measurement() as cf_time_s:
                cfs: Tensor[S, B, H] = explainer.get_cfs(test_x, target)

            nb_steps, nb_paths, _ = cfs.shape

            metrics["time_per_path_step_s"] = cf_time_s.time / (nb_paths * nb_steps)

            cf_preds = classifier.predict(cfs)

            # path_numbers is a non-decreasing vector containing the path number
            # of any step where the predicted class equals the target class.
            # step_numbers contains the step numbers where this is achieved.
            path_numbers, step_numbers = torch.where((target == cf_preds).T)

            valid_path_numbers: Tensor[V] = path_numbers.unique()

            metrics["validity_rate"] = len(valid_path_numbers) / nb_paths

            valid_paths: Tensor[S, V, D] = cfs[:, valid_path_numbers]
            metrics["mean_nll"] = compute_mean_nll(autoencoder_for_nll, valid_paths)

            # Find where the path number changes, which will tell us how many
            # steps were taken to reach the target for each step
            path_ends: Tensor[V] = get_path_ends(cf_preds, target)
            path_mask: Tensor[V, S] = get_path_mask(path_ends, nb_steps).to(
                cf_preds.device
            )
            valid_cf_preds: Tensor[V, S] = cf_preds.T[valid_path_numbers]

            metrics["mean_boundary_crossings"] = compute_mean_boundary_crossings_rate(
                valid_cf_preds, path_mask
            )

            batch_results.append(batchify_metrics(metrics, len(test_x)))

        results = dict(
            pd.DataFrame.from_records(batch_results).sum() / len(data_module.test_set)
        )

        return results
