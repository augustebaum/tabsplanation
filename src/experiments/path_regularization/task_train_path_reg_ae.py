import torch

from config import BLD_MODELS

from experiments.shared.data.task_get_data_module import TaskGetDataModule
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import read, setup, Task
from tabsplanation.explanations.latent_shift import LatentShift
from tabsplanation.explanations.nice_path_regularized import PathRegularizedNICE


class TaskTrainPathRegAe(Task):
    def __init__(self, cfg):
        output_dir = BLD_MODELS
        super(TaskTrainPathRegAe, self).__init__(cfg, output_dir)

        task_get_data_module = TaskGetDataModule(self.cfg.data_module)
        task_train_classifier = TaskTrainModel(
            self.cfg.path_regularized_model.args.classifier
        )
        self.task_deps = [task_get_data_module, task_train_classifier]

        self.depends_on = task_get_data_module.produces
        self.depends_on |= {"classifier": task_train_classifier.produces}

        self.produces |= {
            "model": self.produces_dir / "model.pt",
        }

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        device = setup(cfg.seed)

        data_module = read(depends_on["data_module"])

        classifier = torch.load(depends_on["classifier"]["model"])

        model = PathRegularizedNICE(
            classifier=classifier,
            explainer=LatentShift(
                classifier=classifier,
                autoencoder=None,
                **cfg.path_regularized_model.args.explainer.args
            ),
            autoencoder_args={
                "input_dim": data_module.input_dim,
                **cfg.path_regularized_model.args.autoencoder_args,
            },
        ).to(device)

        model = TaskTrainModel.train_model(data_module, model, cfg)

        torch.save(model, produces["model"])
