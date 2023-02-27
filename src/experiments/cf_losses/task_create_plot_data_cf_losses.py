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
from experiments.shared.data.task_create_cake_on_sea import TaskCreateCakeOnSea
from experiments.shared.data.task_get_data_module import TaskGetDataModule
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import get_data_module, get_object, setup, Task
from tabsplanation.types import Tensor


def get_z0():
    return torch.linspace(-5, 5, steps=25)


def get_x0():
    return torch.linspace(-5, 55, steps=25)


def get_inputs(x0, data_module):
    x: Tensor["nb_points", 2] = torch.cartesian_prod(x0, x0).to(
        data_module.dataset.X.device
    )

    dataset = data_module.dataset

    inputs = dataset.fill_from_2d_point(x)
    normalized_inputs = dataset.normalize(inputs)
    normalized_inputs.requires_grad = True

    return normalized_inputs


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

        task_dataset = TaskGetDataModule.task_dataset(self.cfg.data_module)
        task_train_classifier = TaskTrainModel(self.cfg.classifier)
        task_train_autoencoder = TaskTrainModel(self.cfg.autoencoder)
        self.depends_on = {"dataset": task_dataset.produces}
        self.depends_on |= {"classifier": task_train_classifier.produces}
        self.depends_on |= {"autoencoder": task_train_autoencoder.produces}

        self.produces |= {"results": self.produces_dir / "results.pkl"}

    @classmethod
    def task_function(cls, depends_on, produces, cfg):

        device = setup(cfg.seed)

        z0 = get_z0()
        z: Tensor["nb_points", 2] = torch.cartesian_prod(z0, z0).to(device)

        x0 = get_x0().to(device)

        data_module = TaskGetDataModule.read_data_module(
            depends_on["dataset"], cfg.data_module, device
        )

        normalized_inputs = get_inputs(x0, data_module).to(device)

        classifier = torch.load(depends_on["classifier"]["model"]).to(device)
        autoencoder = torch.load(depends_on["autoencoder"]["model"]).to(device)

        def make_loss(loss_cfg):
            loss_cls = get_object(loss_cfg.class_name)
            return loss_cls() if loss_cfg.args is None else loss_cls(**loss_cfg.args)

        loss_fns = [make_loss(loss_cfg) for loss_cfg in cfg.losses]

        classes = range(data_module.dataset.output_dim)

        result = {
            "x_losses": TaskCreatePlotDataCfLosses.losses_x(
                loss_fns,
                classes,
                classifier,
                normalized_inputs,
            ),
            "z_losses": TaskCreatePlotDataCfLosses.losses_z(
                loss_fns, classes, classifier, autoencoder, z
            ),
            "latent_space_map": TaskCreatePlotDataCfLosses.latent_space_map(
                classifier, autoencoder, normalized_inputs
            ),
            "x_gradients": TaskCreatePlotDataCfLosses.x_gradients(
                loss_fns, classes, classifier, normalized_inputs
            ),
            "z_gradients": TaskCreatePlotDataCfLosses.z_gradients(
                loss_fns, classes, classifier, autoencoder, normalized_inputs
            ),
        }

        with open(produces["results"], "wb") as results_file:
            pickle.dump(result, results_file)

    @staticmethod
    def losses_x(loss_fns, classes, classifier, normalized_inputs):
        x = normalized_inputs.clone()
        f_x = classifier(x)

        def fn(loss_fn, class_):
            return loss_fn(
                f_x,
                source=f_x.argmax(dim=-1),
                target=torch.full((len(f_x),), class_).to(f_x.device),
            )

        return {
            str(loss_fn): {class_: fn(loss_fn, class_) for class_ in classes}
            for loss_fn in loss_fns
        }

    @staticmethod
    def losses_z(loss_fns, classes, classifier, autoencoder, z):
        x_z = autoencoder.decode(z)
        f_x_z = classifier(x_z)

        def fn(loss_fn, class_):
            return loss_fn(
                f_x_z,
                source=f_x_z.argmax(dim=-1),
                target=torch.full((len(f_x_z),), class_).to(f_x_z.device),
            )

        return {
            str(loss_fn): {class_: fn(loss_fn, class_) for class_ in classes}
            for loss_fn in loss_fns
        }

    @staticmethod
    def latent_space_map(classifier, autoencoder, normalized_inputs):
        x = normalized_inputs.clone()

        return {
            "z": autoencoder.encode(x),
            "class": classifier.predict(x),
        }

    @staticmethod
    def x_gradients(loss_fns, classes, classifier, normalized_inputs):
        """Return the opposite of the classifier gradient with respect to `x`."""
        x = normalized_inputs

        losses_x: ResultDict[Loss] = TaskCreatePlotDataCfLosses.losses_x(
            loss_fns, classes, classifier, x
        )

        def fn(loss_fn, class_):
            return -grad(losses_x[str(loss_fn)][class_], x)

        return {
            str(loss_fn): {class_: fn(loss_fn, class_) for class_ in classes}
            for loss_fn in loss_fns
        }

    @staticmethod
    def z_gradients(loss_fns, classes, classifier, autoencoder, normalized_inputs):
        """Return the opposite of the classifier gradient with respect to `z`, mapped
        back to the input space."""
        x = normalized_inputs.clone()
        z_x = autoencoder.encode(x)
        x_z = autoencoder.decode(z_x)
        f_x_z = classifier(x_z)
        original_pred = classifier(x).argmax(dim=-1)

        def fn(loss_fn, class_):
            grad_z = grad(
                loss_fn(
                    f_x_z,
                    source=original_pred,
                    target=torch.full((len(x),), class_).to(f_x_z.device),
                ),
                z_x,
            )

            # Perturb the latents along the negative gradients
            x_tilde = autoencoder.decode(z_x - grad_z)
            return x_tilde - x

        return {
            str(loss_fn): {class_: fn(loss_fn, class_) for class_ in classes}
            for loss_fn in loss_fns
        }
