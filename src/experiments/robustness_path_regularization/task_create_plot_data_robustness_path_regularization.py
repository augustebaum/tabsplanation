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
from tabsplanation.explanations.losses import get_path_mask, MaxPointLoss
from tabsplanation.explanations.nice_path_regularized import random_targets_like
from tabsplanation.metrics import time_measurement
from tabsplanation.types import B, D, S, Tensor, V


def get_mean_distance(data_module, autoencoder, classifier, cfs, target):
    return MaxPointLoss(data_module.train_data[0])(
        autoencoder,
        classifier,
        autoencoder.encode(cfs),
        None,
        target,
    )


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
            "Mean distance to point": result["mean_distance"],
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
                cfs: Tensor[S, B, D] = explainer.get_cfs(test_x, target)

            nb_steps, nb_paths, _ = cfs.shape

            metrics["time_per_path_step_s"] = cf_time_s.time / (nb_paths * nb_steps)

            cf_preds: Tensor[S, B] = classifier.predict(cfs)

            # path_numbers is a non-decreasing vector containing the path number
            # of any step where the predicted class equals the target class.
            # step_numbers contains the step numbers where this is achieved.
            path_numbers, step_numbers = torch.where((target == cf_preds).T)

            valid_path_numbers: Tensor[V] = path_numbers.unique()
            metrics["validity_rate"] = len(valid_path_numbers) / nb_paths

            path_mask: Tensor[S, B] = get_path_mask(cf_preds, target)
            metrics["mean_nll"] = compute_mean_nll(autoencoder_for_nll, cfs, path_mask)

            metrics["mean_distance"] = get_mean_distance(
                data_module, autoencoder, classifier, cfs, target
            )

            batch_results.append(batchify_metrics(metrics, len(test_x)))

        results = dict(
            pd.DataFrame.from_records(batch_results).sum() / len(data_module.test_set)
        )

        return results
