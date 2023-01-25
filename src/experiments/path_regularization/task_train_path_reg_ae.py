import torch

from config import BLD_MODELS
from experiments.shared.task_create_cake_on_sea import TaskCreateCakeOnSea
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import define_task, get_data_module, setup, Task
from tabsplanation.explanations.latent_shift import LatentShift
from tabsplanation.explanations.nice_path_regularized import PathRegularizedNICE


class TaskTrainPathRegAe(Task):
    def __init__(self, cfg):
        output_dir = BLD_MODELS
        super(TaskTrainPathRegAe, self).__init__(cfg, output_dir)

        task_create_cake_on_sea = TaskCreateCakeOnSea(self.cfg)
        task_train_classifier = TaskTrainModel(
            self.cfg.path_regularized_model.args.classifier
        )

        self.depends_on = task_create_cake_on_sea.produces
        self.depends_on |= {"classifier": task_train_classifier.produces}

        self.produces = {
            "model": self.produces_dir / "model.pt",
            "config": self.produces_dir / "config.yaml",
        }

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        device = setup(cfg.seed)

        data_module = get_data_module(depends_on, cfg, device)

        classifier = torch.load(depends_on["classifier"]["model"])

        model = PathRegularizedNICE(
            classifier=classifier,
            explainer=LatentShiftNew(**cfg.path_regularized_model.args.explainer.args),
            autoencoder_args=cfg.path_regularized_model.args.autoencoder_args,
        )

        model = TaskTrainModel.train_model(data_module, model, cfg)

        torch.save(model, produces["model"])


task, task_definition = define_task("path_reg", TaskTrainPathRegAe)
exec(task_definition)
