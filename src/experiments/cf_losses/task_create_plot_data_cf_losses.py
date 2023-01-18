"""Analyze different loss functions that can be used to make the CFX change the
predicted class correctly.

1. For an array of inputs `x` covering the input space, show what the loss is.
2. For an array of latents `z` covering the latent space, show what the loss is
for the decoded points. Show where the classes have been mapped.
3. In all cases, show the gradient of the loss in latent space.
"""
import pickle
from typing import Dict, TypedDict, TypeVar

import torch

from config import BLD_PLOT_DATA
from experiments.shared.task_create_cake_on_sea import TaskCreateCakeOnSea
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import define_task, get_data_module, Task
from tabsplanation.types import Tensor


def get_z0():
    return torch.linspace(-5, 5, steps=25)


def get_x0():
    return torch.linspace(-5, 55, steps=25)


T = TypeVar("T")
LossName = str
ClassNumber = int

ResultDict = Dict[LossName, Dict[ClassNumber, T]]


class Gradients(TypedDict):
    """Input to `numpy.streamplot`."""

    x: Tensor["nb_steps", "nb_steps"]
    y: Tensor["nb_steps", "nb_steps"]
    u: Tensor["nb_steps", "nb_steps"]
    v: Tensor["nb_steps", "nb_steps"]


class LatentSpaceMap(TypedDict):
    z: Tensor["nb_points", "latent_dim"]
    class_: Tensor["nb_points", int]


Loss = Tensor[float]


class CfLossesResult(TypedDict):

    x_losses: ResultDict[Loss]
    z_losses: ResultDict[Loss]
    latent_space_map: LatentSpaceMap
    x_gradients: ResultDict[Gradients]
    z_gradients: ResultDict[Gradients]


class TaskCreatePlotDataCfLosses(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOT_DATA / "cf_losses"
        super(TaskCreatePlotDataCfLosses, self).__init__(cfg, output_dir)

        task_create_cake_on_sea = TaskCreateCakeOnSea(self.cfg)
        task_train_classifier = TaskTrainModel(self.cfg.classifier)
        task_train_autoencoder = TaskTrainModel(self.cfg.autoencoder)
        self.depends_on = task_create_cake_on_sea.produces
        self.depends_on |= {"classifier": task_train_classifier.produces}
        self.depends_on |= {"autoencoder": task_train_autoencoder.produces}

        self.produces |= {"results": self.produces_dir / "results.pkl"}

    @classmethod
    def task_function(cls, depends_on, produces, cfg):

        z0 = get_z0()
        z: Tensor["nb_points", 2] = torch.cartesian_prod(z0, z0)

        x0 = get_x0()
        x: Tensor["nb_points", 2] = torch.cartesian_prod(x0, x0)

        data_module = get_data_module(depends_on, cfg, "cpu")
        dataset = data_module.dataset

        inputs = dataset.fill_from_2d_point(x)
        normalized_inputs = dataset.normalize(inputs)

        classifier = torch.load(depends_on["classifier"]["model"])
        autoencoder = torch.load(depends_on["autoencoder"]["model"])

        # TODO read the classes from config
        from tabsplanation.explanations.losses import (
            AwayLoss,
            BabyStretchLoss,
            StretchLoss,
        )

        def compute_losses(model_logits):
            """Given some model outputs, compute the loss for all choices of target
            class and loss function.

            For each data points the source class is taken to be the predicted class for
            the given logit.
            """
            nb_classes = 3
            return {
                loss.__name__: {
                    class_: loss()(
                        model_logits,
                        source=model_logits.argmax(dim=-1),
                        target=torch.full((len(model_logits),), class_),
                    )
                    for class_ in range(nb_classes)
                }
                for loss in [AwayLoss, BabyStretchLoss, StretchLoss]
            }

        result = {}
        f_x = classifier(normalized_inputs)
        result["x_losses"] = compute_losses(f_x)
        # 2.a.
        result["z_losses"] = compute_losses(classifier(autoencoder.decode(z)))
        # 2.b.
        result["latent_space_map"] = {
            "z": autoencoder.encode(normalized_inputs),
            "class": f_x.argmax(dim=-1),
        }

        # 3.a. Gradients of the losses, taken with respect to x
        # result["x_gradients"] =  {
        #         loss.__name__: {
        #             class_: torch.autograd.grad(result["x_losses"][loss.__name__][class_], x)[0]
        #             for class_ in range(nb_classes)
        #         }
        #         for loss in [AwayLoss, BabyStretchOutLoss, StretchOutLoss]
        #     }
        # 3.a. Gradients of the losses, taken with respect to z, mapped back to the input space
        # result["z_gradients"] =

        with open(produces["results"], "wb") as results_file:
            pickle.dump(result, results_file)


task, task_definition = define_task("cf_losses", TaskCreatePlotDataCfLosses)
exec(task_definition)
