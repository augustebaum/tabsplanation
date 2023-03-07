import random

import pandas as pd
import torch
from omegaconf import OmegaConf
from torcheval.metrics.functional import auc

from config import BLD_PLOT_DATA
from experiments.path_regularization_cake_on_sea.task_train_path_regularized_ae import (
    TaskTrainPathRegAe,
)
from experiments.shared.data.task_get_data_module import TaskGetDataModule
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import clone_config, get_object, read, setup, Task, write

from tabsplanation.explanations.nice_path_regularized import random_targets_like
from tabsplanation.metrics import lof, time_measurement, train_lof
from tabsplanation.types import B, H, S, Tensor


def where_changes(tensor: Tensor[B, S], dim=-1) -> Tensor[B, S]:
    """Return a Tensor containing 1 if the corresponding entry is
    different from the previous one (according to `dim`), 0 otherwise.

    By convention the result always starts with 0.
    """
    prepend = tensor.index_select(dim, torch.tensor([0]).to(tensor.device))
    diffs = tensor.diff(prepend=prepend, dim=dim)
    result = torch.zeros_like(tensor).to(tensor.device)
    result[diffs != 0] = 1
    return result


def indices_where_changes(tensor: Tensor[S]) -> Tensor[S]:
    """Return a Tensor containing the indices in `tensor` where the entry is
    different from the previous one.

    By convention the result always starts with 0.
    """
    if len(tensor) == 0:
        return tensor
    zero = torch.tensor([0]).to(tensor.device)
    return torch.cat([zero, where_changes(tensor).nonzero().squeeze()])


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

        results = []

        for data_module_name, values in depends_on.items():
            device = setup(cfg.seed)

            classifier_cfg = OmegaConf.load(values["classifier"]["full_config"])

            data_module = TaskGetDataModule.read_data_module(
                values["dataset"], classifier_cfg.data_module, "cpu"
            )

            print(f"Dataset: {data_module.dataset.__class__.__name__}")

            classifier = read(values["classifier"]["model"], device=device)

            for (seed, path_regularized), autoencoder_path in values[
                "autoencoders"
            ].items():
                print(f"Seed: {seed}")
                autoencoder = read(autoencoder_path["model"], device=device)

                for path_method in cfg.explainers:

                    for loss_fn in cfg.losses:
                        path_results = (
                            TaskCreatePlotDataPathRegularization.get_path_results(
                                data_module,
                                classifier,
                                autoencoder,
                                loss_fn,
                                path_method,
                            )
                        )

                        result = {
                            "data_module": data_module_name,
                            "path_method": path_method,
                            "seed": seed,
                            "path_regularized": path_regularized,
                            "loss": loss_fn,
                            **path_results,
                        }

                        results.append(result)

        def parse_result(result):
            get_object_name = lambda s: parse_full_qualified_object(s)[1]
            return {
                "Dataset": get_object_name(result["data_module"]).removesuffix(
                    "Dataset"
                ),
                "Path method": result["path_method"]["name"],
                "Path regularization": result["path_regularized"],
                "Loss function": result["loss"]["name"],
                "validity_rate (%)": result["validity_rate"] * 100,
                "AUC of LOF": result["auc_lof"],
                r"\Delta t (ns)": result["time_per_iteration_s"] * (10 ** 9),
                "Mean #BC": result["mean_boundary_crossings"],
            }

        results = [parse_result(result) for result in results]

        write(results, produces["results"])

    @staticmethod
    def get_path_results(data_module, classifier, autoencoder, loss_fn, explainer):
        torch.cuda.empty_cache()

        # import pdb

        # pdb.set_trace()
        trained_lof = train_lof(data_module.dataset.X)

        loss_fn = get_object(loss_fn.class_name)()

        explainer_cls = get_object(explainer.class_name)
        explainer_hparams = explainer.args.hparams

        explainer = explainer_cls(classifier, autoencoder, explainer_hparams, loss_fn)

        batch_results = []
        total_nb_points = 0

        for test_x, _ in data_module.test_dataloader(batch_size=4_000):
            # test_x = data_module.test_data[0][:5_000].to(classifier.device)
            test_x = test_x.to(classifier.device)

            y_predict = classifier.predict(test_x)
            target = random_targets_like(y_predict, data_module.dataset.output_dim)

            with time_measurement() as cf_time_s:
                cfs: Tensor[S, B, H] = explainer.get_cfs(test_x, target)

            nb_steps, nb_paths, _ = cfs.shape

            # Time for one step of one path: Done!
            time_per_path_step_s = cf_time_s.time / (nb_paths * nb_steps)

            cf_preds = classifier.predict(cfs)

            """We only count the AUC until the target class is first reached.
            If the target class is never reached, the path is skipped."""

            # path_numbers is a non-decreasing vector containing the path number
            # of any step where the predicted class equals the target class.
            # step_numbers contains the step numbers where this is achieved.
            path_numbers, step_numbers = torch.where((target == cf_preds).T)
            valid_path_numbers = path_numbers.unique()
            nb_valid = len(valid_path_numbers)

            # Validity rate: Done!
            validity_rate = nb_valid / nb_paths

            valid_cf_preds: Tensor[nb_valid, S] = cf_preds.T[valid_path_numbers]

            # Find where the path number changes, which will tell us how many
            # steps were taken to reach the target for each step

            if nb_valid == 0:
                mean_auc_lof = 0
            else:
                path_ends: Tensor[nb_valid] = step_numbers[
                    indices_where_changes(path_numbers)
                ]

                xs: Tensor[nb_valid, S] = torch.stack(
                    [
                        torch.cat(
                            [
                                torch.linspace(0, 1, steps=path_end),
                                torch.ones(nb_steps - path_end),
                            ]
                        )
                        for path_end in path_ends
                    ]
                ).to(cf_preds.device)

                lofs: Tensor[nb_valid, S] = lof(
                    trained_lof, cfs[:, valid_path_numbers]
                ).T

                auc_lofs: Tensor[nb_valid] = auc(xs, lofs)
                # AUC: Done!
                mean_auc_lof = auc_lofs.mean().item()

            if nb_valid == 0:
                mean_boundary_crossings_rate = 0
            else:
                # 1 until the path reaches the target, then 0
                path_mask: Tensor[nb_valid, S] = torch.stack(
                    [
                        torch.cat(
                            [
                                torch.ones(path_end),
                                torch.zeros(nb_steps - path_end),
                            ]
                        )
                        for path_end in path_ends
                    ]
                ).to(cf_preds.device)

                bound_crossings_before_target: Tensor[nb_valid, S] = (
                    where_changes(valid_cf_preds) * path_mask
                )
                mean_boundary_crossings_rate = (
                    bound_crossings_before_target.sum(dim=1).mean().item()
                )

            batch_size = len(test_x)
            batch_results.append(
                {
                    "auc_lof": mean_auc_lof * batch_size,
                    "time_per_iteration_s": time_per_path_step_s * batch_size,
                    "validity_rate": validity_rate * batch_size,
                    "mean_boundary_crossings": mean_boundary_crossings_rate
                    * batch_size,
                }
            )
            total_nb_points += batch_size

        results = dict(pd.DataFrame.from_records(batch_results).sum() / total_nb_points)

        return results
