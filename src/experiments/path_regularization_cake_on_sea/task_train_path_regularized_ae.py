import torch

from config import BLD_MODELS

from experiments.shared.data.task_get_data_module import TaskGetDataModule
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import get_object, read, setup, Task
from tabsplanation.explanations.losses import MaxPointLoss
from tabsplanation.explanations.nice_path_regularized import PathRegularizedNICE


class TaskTrainPathRegAe(Task):
    def __init__(self, cfg):
        output_dir = BLD_MODELS
        super(TaskTrainPathRegAe, self).__init__(cfg, output_dir)

        task_dataset = TaskGetDataModule.task_dataset(self.cfg.data_module)

        task_train_classifier = TaskTrainModel(self.cfg.model.args.classifier)
        self.task_deps = [task_dataset, task_train_classifier]

        self.depends_on = {
            "dataset": task_dataset.produces,
            "classifier": task_train_classifier.produces,
        }

        self.produces |= {
            "model": self.produces_dir / "model.pt",
        }

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        device = setup(cfg.seed)

        data_module = TaskGetDataModule.read_data_module(
            depends_on["dataset"], cfg.data_module, device
        )

        classifier = read(depends_on["classifier"]["model"], device=device)

        model = initialize_model(cfg.model.args, classifier, data_module)
        model = TaskTrainModel.train_model(data_module, model, cfg)

        torch.save(model, produces["model"])


def initialize_model(model_args, classifier, data_module):
    path_loss_cls = get_object(model_args.path_loss_fn.class_name)
    if path_loss_cls == MaxPointLoss:
        train_inputs, _ = data_module.train_data
        path_loss_fn = path_loss_cls(train_inputs)
    else:
        path_loss_fn = path_loss_cls()

    return PathRegularizedNICE(
        classifier=classifier,
        explainer_cls=get_object(model_args.explainer.class_name),
        explainer_hparams=model_args.explainer.args.hparams,
        autoencoder_args={
            "input_dim": data_module.input_dim,
            **model_args.autoencoder_args,
        },
        hparams=model_args.hparams,
        path_loss_fn=path_loss_fn,
    ).to(classifier.device)
