"""Compare paths generated by several methods.

The goal is to determine visually if a path-regularized latent space gives rise to paths
that fit constraints better than those produced with a non-regularized latent space.
"""

import torch

from config import BLD_PLOT_DATA
from experiments.cf_losses.task_create_plot_data_cf_losses import (
    get_inputs,
    get_x0,
    TaskCreatePlotDataCfLosses,
)
from experiments.path_regularization.task_train_path_reg_ae import TaskTrainPathRegAe
from experiments.shared.data.task_create_cake_on_sea import TaskCreateCakeOnSea
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import (
    define_task,
    get_data_module,
    read,
    setup,
    Task,
    write,
)
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

        device = setup(cfg.seed)

        classifier = read(depends_on["classifier"]["model"], device)
        autoencoder = read(depends_on["autoencoder"]["model"], device)
        path_regularized_autoencoder = read(
            depends_on["path_regularized_autoencoder"]["model"], device
        )

        results = {}

        # 0. Get some input points
        x0 = get_x0().to(device)
        data_module = get_data_module(depends_on, cfg, device)
        normalized_inputs = get_inputs(x0, data_module).to(device)

        # 1. Plot the latent spaces
        results |= {
            "unregularized_latent_space_map": TaskCreatePlotDataCfLosses.latent_space_map(
                classifier, autoencoder, normalized_inputs
            ),
            "path_regularized_latent_space_map": TaskCreatePlotDataCfLosses.latent_space_map(
                classifier, path_regularized_autoencoder, normalized_inputs
            ),
        }

        # 2. Show a few paths
        latent_shift_hparams = {"shift_step": 0.005, "max_iter": 100}
        revise_hparams = {
            "optimizer": "adam",
            "lr": 0.1,
            "max_iter": 100,
            "distance_regularization": 0.5,
        }

        path_methods = {
            "Latent shift": LatentShift(classifier, autoencoder, latent_shift_hparams),
            "Revise": Revise(classifier, autoencoder, revise_hparams),
            "Latent shift with path regularization": LatentShift(
                classifier, path_regularized_autoencoder, latent_shift_hparams
            ),
        }

        # Input that should be predicted to be class 0
        input = torch.tensor(
            [
                [25.0, 10.0],
            ]
        ).to(device)
        normalized_input = data_module.dataset.normalize(input)
        target_class = 0 if classifier.predict(normalized_input) == 2 else 2
        target_class = torch.tensor([target_class]).to(device)

        results["paths"] = {}
        for method_name, method in path_methods.items():
            path = method.get_counterfactuals(normalized_input, target_class)
            path.explained_input.input = input
            path.xs = data_module.dataset.normalize_inverse(path.xs)
            results["paths"][method_name] = path

        write(results, produces["results"])


task, task_definition = define_task("path_reg", TaskCreatePlotDataPathRegularization)
exec(task_definition)
