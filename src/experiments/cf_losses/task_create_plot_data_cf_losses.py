import pickle

import torch

from config import BLD_PLOT_DATA
from experiments.shared.task_create_cake_on_sea import TaskCreateCakeOnSea
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import define_task, get_data_module, setup, Task
from tabsplanation.types import Tensor


def get_z0(cfg):
    return torch.linspace(cfg.lo, cfg.hi, steps=cfg.nb_steps)


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

        # device = setup(cfg.seed)

        z0 = get_z0(cfg)
        z: Tensor["nb_points", 2] = torch.cartesian_prod(z0, z0)

        # data_module = get_data_module(depends_on, cfg, device)
        # dataset = data_module.dataset

        # inputs = dataset.fill_from_2d_point(inputs_x)
        # normalized_inputs = dataset.normalize(inputs)

        classifier = torch.load(depends_on["classifier"]["model"])
        autoencoder = torch.load(depends_on["autoencoder"]["model"])

        from tabsplanation.explanations.losses import (
            AwayLoss,
            BabyStretchOutLoss,
            StretchOutLoss,
        )

        model_logits = classifier(autoencoder.decode(z))
        source = model_logits.argmax(dim=-1)

        # torch doesn't seem to have a more convenient map function
        # source_to_target = {0: 1, 1: 2, 2: 0}
        # target = torch.zeros(len(source), dtype=int)
        # target.map_(source, lambda _, s: source_to_target.get(s))

        nb_classes = 3
        # result = {"inputs": inputs, "latents": autoencoder.encode(normalized_inputs)}
        result = {
            loss.__name__: {
                class_: loss()(
                    model_logits,
                    source=source,
                    target=torch.full((len(source),), class_),
                )
                for class_ in range(nb_classes)
            }
            for loss in [AwayLoss, BabyStretchOutLoss, StretchOutLoss]
        }

        with open(produces["results"], "wb") as results_file:
            pickle.dump(result, results_file)


task, task_definition = define_task("cf_losses", TaskCreatePlotDataCfLosses)
exec(task_definition)
