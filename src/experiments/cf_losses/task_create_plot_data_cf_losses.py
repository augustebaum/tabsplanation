"""Analyze different loss functions that can be used to make the CFX change the
predicted class correctly.

1. For an array of inputs `x` covering the input space, show what the loss is.
2. For an array of latents `z` covering the latent space, show what the loss is
for the decoded points. Show where the classes have been mapped.
3. In all cases, show the gradient of the loss in latent space.
"""
import pickle
from typing import Any, Callable, Dict, Iterable, TypedDict, TypeVar

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
LossClass = Any
ClassNumber = int


class ResultDict:
    def __class_getitem__(cls, item: Any):
        """To make `ResultDict[T]` work as a type annotation."""
        return Dict[LossName, Dict[ClassNumber, item]]

    def new(
        fn: Callable[[LossClass, ClassNumber], T],
        loss_classes: Iterable[LossClass],
        classes: Iterable[ClassNumber],
    ):
        return {
            loss.__name__: {class_: fn(loss, class_) for class_ in classes}
            for loss in loss_classes
        }

    def from_fn(loss_classes, classes):
        def x(fn):
            return ResultDict.new(fn, loss_classes, classes)

        return x


# class Gradients(TypedDict):
#     """Input to `numpy.streamplot`."""

#     x: Tensor["nb_steps", "nb_steps"]
#     y: Tensor["nb_steps", "nb_steps"]
#     u: Tensor["nb_steps", "nb_steps"]
#     v: Tensor["nb_steps", "nb_steps"]

Gradients = Tensor["nb_steps_squared ** 2", 2]


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


def grad(losses, inputs):
    return torch.autograd.grad(losses.sum(), inputs, retain_graph=True)[0]


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
        normalized_inputs.requires_grad = True

        classifier = torch.load(depends_on["classifier"]["model"])
        autoencoder = torch.load(depends_on["autoencoder"]["model"])

        # TODO read the classes from config
        from tabsplanation.explanations.losses import (
            AwayLoss,
            BabyStretchLoss,
            StretchLoss,
        )

        loss_classes = [
            AwayLoss,
            BabyStretchLoss,
            StretchLoss,
        ]
        nb_classes = 3
        classes = range(nb_classes)

        result = {
            "x_losses": TaskCreatePlotDataCfLosses.losses_x(
                loss_classes,
                classes,
                classifier,
                normalized_inputs,
            ),
            "z_losses": TaskCreatePlotDataCfLosses.losses_z(
                loss_classes, classes, classifier, autoencoder, z
            ),
            "latent_space_map": TaskCreatePlotDataCfLosses.latent_space_map(
                classifier, autoencoder, normalized_inputs
            ),
            "x_gradients": TaskCreatePlotDataCfLosses.x_gradients(
                loss_classes, classes, classifier, normalized_inputs
            ),
            "z_gradients": TaskCreatePlotDataCfLosses.z_gradients(
                loss_classes, classes, classifier, autoencoder, normalized_inputs
            ),
        }

        with open(produces["results"], "wb") as results_file:
            pickle.dump(result, results_file)

    @staticmethod
    def losses_x(loss_classes, classes, classifier, normalized_inputs):
        x = normalized_inputs.clone()
        f_x = classifier(x)

        def fn(loss, class_):
            return loss()(
                f_x,
                source=f_x.argmax(dim=-1),
                target=torch.full((len(f_x),), class_),
            )

        return {
            loss.__name__: {class_: fn(loss, class_) for class_ in classes}
            for loss in loss_classes
        }

    @staticmethod
    def losses_z(loss_classes, classes, classifier, autoencoder, z):
        x_z = autoencoder.decode(z)
        f_x_z = classifier(x_z)

        def fn(loss, class_):
            return loss()(
                f_x_z,
                source=f_x_z.argmax(dim=-1),
                target=torch.full((len(f_x_z),), class_),
            )

        return {
            loss.__name__: {class_: fn(loss, class_) for class_ in classes}
            for loss in loss_classes
        }

    @staticmethod
    def latent_space_map(classifier, autoencoder, normalized_inputs):
        x = normalized_inputs.clone()

        return {
            "z": autoencoder.encode(x),
            "class": classifier.predict(x),
        }

    @staticmethod
    def x_gradients(loss_classes, classes, classifier, normalized_inputs):
        """Return the opposite of the classifier gradient with respect to `x`."""
        x = normalized_inputs

        losses_x: ResultDict[Loss] = TaskCreatePlotDataCfLosses.losses_x(
            loss_classes, classes, classifier, x
        )

        def fn(loss, class_):
            return -grad(losses_x[loss.__name__][class_], x)

        return {
            loss.__name__: {class_: fn(loss, class_) for class_ in classes}
            for loss in loss_classes
        }

    @staticmethod
    def z_gradients(loss_classes, classes, classifier, autoencoder, normalized_inputs):
        """Return the opposite of the classifier gradient with respect to `z`, mapped
        back to the input space."""
        x = normalized_inputs.clone()
        z_x = autoencoder.encode(x)
        x_z = autoencoder.decode(z_x)
        f_x_z = classifier(x_z)
        original_pred = classifier(x).argmax(dim=-1)

        def fn(loss, class_):
            grad_z = grad(
                loss()(
                    f_x_z,
                    source=original_pred,
                    target=torch.full((len(x),), class_),
                ),
                z_x,
            )

            # Perturb the latents along the negative gradients
            x_tilde = autoencoder.decode(z_x - grad_z)
            return x_tilde - x

        return {
            loss.__name__: {class_: fn(loss, class_) for class_ in classes}
            for loss in loss_classes
        }


task, task_definition = define_task("cf_losses", TaskCreatePlotDataCfLosses)
exec(task_definition)
