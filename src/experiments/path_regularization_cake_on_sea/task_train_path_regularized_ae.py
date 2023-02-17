import torch

from config import BLD_MODELS

from experiments.shared.data.task_get_data_module import TaskGetDataModule
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import get_object, read, setup, Task
from tabsplanation.explanations.nice_path_regularized import PathRegularizedNICE


class TaskTrainPathRegAe(Task):
    def __init__(self, cfg):
        output_dir = BLD_MODELS
        super(TaskTrainPathRegAe, self).__init__(cfg, output_dir)

        task_dataset = TaskGetDataModule.task_dataset(self.cfg.data_module)

        task_train_classifier = TaskTrainModel(self.cfg.model.args.classifier)
        self.task_deps = [task_dataset, task_train_classifier]

        self.depends_on = task_dataset.produces
        self.depends_on |= {"classifier": task_train_classifier.produces}

        self.produces |= {
            "model": self.produces_dir / "model.pt",
        }

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        device = setup(cfg.seed)

        data_module = TaskGetDataModule.read_data_module(
            depends_on, cfg.data_module, device
        )

        classifier = read(depends_on["classifier"]["model"], device=device)

        explainer_cls = get_object(cfg.model.args.explainer.class_name)

        model = PathRegularizedNICE(
            classifier=classifier,
            explainer_cls=explainer_cls,
            explainer_hparams=cfg.model.args.explainer.args.hparams,
            autoencoder_args={
                "input_dim": data_module.input_dim,
                **cfg.model.args.autoencoder_args,
            },
        ).to(device)

        model = TaskTrainModel.train_model(data_module, model, cfg)

        torch.save(model, produces["model"])
