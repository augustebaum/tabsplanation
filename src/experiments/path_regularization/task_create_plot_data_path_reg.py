"""Compare paths generated by several methods.

The goal is to determine visually if a path-regularized latent space gives rise to paths
that fit constraints better than those produced with a non-regularized latent space.
"""

from config import BLD_PLOT_DATA
from experiments.cf_losses.task_create_plot_data_cf_losses import (
    TaskCreatePlotDataCfLosses,
)
from experiments.path_regularization.task_train_path_reg_ae import TaskTrainPathRegAe
from experiments.shared.task_create_cake_on_sea import TaskCreateCakeOnSea
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import define_task, Task, write
from tabsplanation.explanations.latent_shift import LatentShift
from tabsplanation.explanations.revise import Revise


class TaskCreatePlotDataPathRegularization(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOT_DATA / "path_reg"
        super(TaskCreatePlotDataPathRegularization, self).__init__(cfg, output_dir)

        task_create_cake_on_sea = TaskCreateCakeOnSea(self.cfg)

        task_train_classifier = TaskTrainModel(self.cfg.classifier)
        task_train_autoencoder = TaskTrainModel(self.cfg.autoencoder)

        task_train_path_regularized_autoencoder = TaskTrainPathRegAe(self.cfg)
        self.depends_on = task_create_cake_on_sea.produces
        # self.depends_on |= {"classifier": task_train_classifier.produces}
        self.depends_on |= {
            "path_regularized_autoencoder": task_train_path_regularized_autoencoder.produces,
            "autoencoder": task_train_autoencoder.produces,
            "classifier": task_train_classifier.produces,
        }

        self.produces |= {"results": self.produces_dir / "results.pkl"}

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        classifier = depends_on["classifier"]["model"]
        path_regularized_autoencoder = depends_on["path_regularized_autoencoder"][
            "model"
        ]
        autoencoder = depends_on["autoencoder"]["model"]

        results = {}

        # 1. Plot the latent spaces
        results |= {
            "unregularized_latent_space_map": TaskCreatePlotDataCfLosses.latent_space_map(
                classifier, autoencoder, normalized_inputs
            ),
            "path_regularized_latent_space_map": TaskCreatePlotDataCfLosses.latent_space_map(
                classifier, path_regularized_autoencoder, normalized_inputs
            ),
        }

        latent_shift_hparams = {"shift_step": 0.005, "max_iter": 100}
        revise_hparams = {
            "optimizer": "adam",
            "lr": 0.1,
            "max_iter": 100,
            "distance_regularization": 0.5,
        }

        # 2. Show a few paths
        path_methods = [
            LatentShift(classifier, autoencoder, latent_shift_hparams),
            Revise(classifier, autoencoder, revise_hparams),
            LatentShift(classifier, path_regularized_autoencoder, latent_shift_hparams),
        ]

        results["paths"] = {
            method.__class__.__name__: method.get_counterfactuals(input, target_class)
            for method in path_methods
        }

        write(results, produces["results"])
        # for result in results:
        # TaskPlotClass2Paths.plot_path(ax, result["path"])

        # PathRegularizedNICE()


task, task_definition = define_task("path_reg", TaskCreatePlotDataPathRegularization)
exec(task_definition)
