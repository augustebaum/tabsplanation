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
from experiments.shared.utils import get_object_name, Task
from tabsplanation.explanations.losses import get_path_mask

from tabsplanation.explanations.nice_path_regularized import random_targets_like
from tabsplanation.metrics import time_measurement
from tabsplanation.types import B, D, S, Tensor


def compute_mean_distance_to_max(
    autoencoder, cfs: Tensor[S, B, D], path_mask: Tensor[S, B], point: Tensor[D]
):
    s, b, d = cfs.shape
    distances_2d: Tensor[B * S] = torch.linalg.vector_norm(
        cfs.view(-1, d) - point, dim=-1
    )
    distances: Tensor[S, B] = torch.masked.masked_tensor(
        distances_2d.view(s, b), path_mask
    )

    return distances.amin(dim=0).mean().get_data().nan_to_num(torch.tensor(0))


class TaskCreatePlotDataRobustnessPathRegularization(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOT_DATA / "robustness_path_regularization"
        super(TaskCreatePlotDataRobustnessPathRegularization, self).__init__(
            cfg, output_dir
        )
        self.task_deps, self.depends_on = TaskCreatePlotDataPathRegularization.setup(
            cfg
        )
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
            r"Validity rate (\%)": result["validity_rate"] * 100,
            r"\Delta t (ns)": result["time_per_path_step_s"] * (10 ** 9),
            "Mean NLL": result["mean_nll"],
            "Mean distance to max": result["mean_distance_to_max"],
        }

    @staticmethod
    def get_path_results(
        data_module, classifier, autoencoder, loss_fn, explainer, autoencoder_for_nll
    ):
        torch.cuda.empty_cache()

        loss_fn = get_loss_fn(loss_fn)

        explainer = get_explainer(classifier, autoencoder, explainer, loss_fn)

        batch_results = []

        max_point: Tensor[D] = (
            data_module.train_data[0].amax(dim=0).to(classifier.device)
        )

        for test_x, _ in data_module.test_dataloader(batch_size=4_000):
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

            metrics["mean_distance_to_max"] = compute_mean_distance_to_max(
                autoencoder, cfs, path_mask, max_point
            )

            batch_results.append(batchify_metrics(metrics, nb_paths))

        results = dict(
            pd.DataFrame.from_records(batch_results).sum() / len(data_module.test_set)
        )

        return results
